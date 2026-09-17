use super::*;
use std::{
    error::Error as _,
    fs::OpenOptions,
    io::{Seek, SeekFrom, Write},
};

#[test]
fn retained_handle_reads_exact_bytes_without_using_or_moving_a_shared_cursor() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokenizer.json");
    let old = directory.path().join("original.json");
    let bytes: Vec<u8> = (0..140_017).map(|n| (n % 251) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let file = File::open(&path).unwrap();
    // Seal after the rename: a later same-name replacement is another inode.
    std::fs::rename(&path, &old).unwrap();
    let mut alias = file.try_clone().unwrap();
    let read = PreparedArtifactFileRead::new(file).unwrap();
    std::fs::write(&path, b"replacement contents").unwrap();
    alias.seek(SeekFrom::Start(97)).unwrap();
    assert_eq!(read.byte_len(), bytes.len());
    let mut destination = vec![0; bytes.len()];
    read.read_into(&mut destination).unwrap();
    assert_eq!(destination, bytes);
    assert_eq!(alias.stream_position().unwrap(), 97);
    assert_eq!(std::fs::read(path).unwrap(), b"replacement contents");
}

#[test]
fn actual_truncate_grow_and_same_length_rewrite_preserve_the_written_prefix() {
    for mode in 0..3 {
        let mut file = tempfile::tempfile().unwrap();
        let bytes = vec![b'a'; 65_545];
        file.write_all(&bytes).unwrap();
        let mut writer = file.try_clone().unwrap();
        let read = PreparedArtifactFileRead::new(file).unwrap();
        let mut destination = vec![0; bytes.len()];
        let mut changed = false;
        let failure = read
            .read_into_with(&mut destination, |filled| {
                if changed {
                    return;
                }
                assert_eq!(filled, 65_536);
                changed = true;
                match mode {
                    0 => writer.set_len(65_536).unwrap(),
                    1 => {
                        writer.seek(SeekFrom::End(0)).unwrap();
                        writer.write_all(b"growth").unwrap();
                    }
                    _ => {
                        writer.seek(SeekFrom::Start(65_536)).unwrap();
                        writer.write_all(b"different").unwrap();
                        let changed = writer.metadata().unwrap().modified().unwrap()
                            + std::time::Duration::from_secs(1);
                        writer
                            .set_times(std::fs::FileTimes::new().set_modified(changed))
                            .unwrap();
                    }
                }
            })
            .unwrap_err();
        assert!(changed);
        assert_eq!(&destination[..65_536], &bytes[..65_536]);
        match mode {
            0 => {
                assert!(matches!(failure.cause(), ArtifactFileReadError::Truncated));
                assert_eq!(failure.filled_bytes(), 65_536);
                assert_eq!(&destination[65_536..], &[0; 9]);
            }
            1 => {
                assert!(matches!(failure.cause(), ArtifactFileReadError::Grown));
                assert_eq!(failure.filled_bytes(), bytes.len());
            }
            _ => {
                assert!(matches!(failure.cause(), ArtifactFileReadError::Changed));
                assert_eq!(&destination[65_536..], b"different");
            }
        }
    }
}

#[test]
fn changed_preflight_wrong_destination_and_real_os_read_error_do_not_publish_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.json");
    std::fs::write(&path, b"nonzero").unwrap();
    let read = PreparedArtifactFileRead::new(File::open(&path).unwrap()).unwrap();
    std::fs::write(&path, b"changed length").unwrap();
    let mut destination = [17; 7];
    let failure = read.read_into(&mut destination).unwrap_err();
    assert!(matches!(failure.cause(), ArtifactFileReadError::Changed));
    assert_eq!(failure.filled_bytes(), 0);
    assert_eq!(destination, [17; 7]);
    let read = PreparedArtifactFileRead::new(File::open(&path).unwrap()).unwrap();
    let failure = read.read_into(&mut destination).unwrap_err();
    assert!(matches!(
        failure.cause(),
        ArtifactFileReadError::DestinationLength
    ));
    assert_eq!(failure.filled_bytes(), 0);
    let read =
        PreparedArtifactFileRead::new(OpenOptions::new().write(true).open(&path).unwrap()).unwrap();
    let mut destination = vec![29; read.byte_len()];
    let failure = read.read_into(&mut destination).unwrap_err();
    let source = failure
        .cause()
        .source()
        .unwrap()
        .downcast_ref::<io::Error>()
        .unwrap();
    assert!(source.raw_os_error().is_some());
    assert_eq!(failure.filled_bytes(), 0);
    assert!(destination.iter().all(|x| *x == 29));
    assert!(matches!(
        PreparedArtifactFileRead::new(File::open(directory.path()).unwrap()),
        Err(ArtifactFileReadError::NotRegular)
    ));
    PreparedArtifactFileRead::new(tempfile::tempfile().unwrap())
        .unwrap()
        .read_into(&mut [])
        .unwrap();
}

#[test]
fn captured_version_rejects_same_bytes_from_another_file_and_later_source_rewrite() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("published.safetensors");
    let foreign = directory.path().join("same-bytes.safetensors");
    let bytes = b"actual nonzero published cache bytes";
    std::fs::write(&path, bytes).unwrap();
    std::fs::write(&foreign, bytes).unwrap();
    let file = File::open(&path).unwrap();
    let version = ArtifactFileVersion::capture(&file).unwrap();
    assert_eq!(version.byte_len(), bytes.len());
    assert!(matches!(
        version.bind(File::open(&foreign).unwrap()),
        Err(ArtifactFileReadError::Changed)
    ));
    let mut output = vec![0; version.byte_len()];
    version
        .bind(File::open(&path).unwrap())
        .unwrap()
        .read_into(&mut output)
        .unwrap();
    assert_eq!(output, bytes);
    let mut writer = OpenOptions::new().write(true).open(&path).unwrap();
    writer.write_all(&vec![b'x'; bytes.len()]).unwrap();
    let modified =
        writer.metadata().unwrap().modified().unwrap() + std::time::Duration::from_secs(1);
    writer
        .set_times(std::fs::FileTimes::new().set_modified(modified))
        .unwrap();
    assert!(matches!(
        version.validate(&file),
        Err(ArtifactFileReadError::Changed)
    ));
    assert!(matches!(
        version.bind(File::open(&path).unwrap()),
        Err(ArtifactFileReadError::Changed)
    ));
    ArtifactFileVersion::control_bytes().unwrap();
}
