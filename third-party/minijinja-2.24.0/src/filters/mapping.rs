//! One ordered filter mapping loop; adapters retain their own source and output.
pub(crate) trait Driver {
    type Item;
    type Error;
    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error>;
    fn transform(&mut self, value: Self::Item) -> Result<Self::Item, Self::Error>;
    fn retain(&mut self, value: Self::Item) -> Result<(), Self::Error>;
}
pub(crate) fn run<D: Driver>(driver: &mut D) -> Result<(), D::Error> {
    while let Some(value) = driver.next()? {
        let result = driver.transform(value)?;
        driver.retain(result)?;
    }
    Ok(())
}
pub(crate) fn ordinary<T, E, I, F>(items: I, output: &mut Vec<T>, transform: F) -> Result<(), E>
where
    I: Iterator<Item = T>,
    F: FnMut(T) -> Result<T, E>,
{
    struct Ordinary<'a, T, I, F> {
        items: I,
        output: &'a mut Vec<T>,
        transform: F,
    }
    impl<T, E, I, F> Driver for Ordinary<'_, T, I, F>
    where
        I: Iterator<Item = T>,
        F: FnMut(T) -> Result<T, E>,
    {
        type Item = T;
        type Error = E;
        fn next(&mut self) -> Result<Option<T>, E> {
            Ok(self.items.next())
        }
        fn transform(&mut self, value: T) -> Result<T, E> {
            (self.transform)(value)
        }
        fn retain(&mut self, value: T) -> Result<(), E> {
            self.output.push(value);
            Ok(())
        }
    }
    run(&mut Ordinary {
        items,
        output,
        transform,
    })
}
pub(crate) fn control_bytes<D: Driver>() -> Option<usize> {
    use std::mem::size_of;
    size_of::<D>()
        .checked_add(size_of::<[D::Item; 2]>())?
        .checked_add(size_of::<Result<Option<D::Item>, D::Error>>())?
        .checked_add(size_of::<Result<D::Item, D::Error>>())?
        .checked_add(size_of::<Result<(), D::Error>>())?
        .checked_add(size_of::<&mut D>())
}
