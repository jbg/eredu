//! One ordered selection loop; storage adapters own iteration and destinations.
pub(crate) trait Driver {
    type Item;
    type Error;
    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error>;
    fn test(&mut self, value: &Self::Item) -> Result<bool, Self::Error>;
    fn retain(&mut self, value: Self::Item) -> Result<(), Self::Error>;
}
pub(crate) fn run<D: Driver>(driver: &mut D, invert: bool) -> Result<(), D::Error> {
    while let Some(value) = driver.next()? {
        if driver.test(&value)? != invert {
            driver.retain(value)?;
        }
    }
    Ok(())
}
pub(crate) fn ordinary<T, E, I, F>(items: I, invert: bool, test: F) -> Result<Vec<T>, E>
where
    I: Iterator<Item = T>,
    F: FnMut(&T) -> Result<bool, E>,
{
    struct Ordinary<T, I, F> {
        items: I,
        test: F,
        output: Vec<T>,
    }
    impl<T, E, I, F> Driver for Ordinary<T, I, F>
    where
        I: Iterator<Item = T>,
        F: FnMut(&T) -> Result<bool, E>,
    {
        type Item = T;
        type Error = E;
        fn next(&mut self) -> Result<Option<T>, E> {
            Ok(self.items.next())
        }
        fn test(&mut self, value: &T) -> Result<bool, E> {
            (self.test)(value)
        }
        fn retain(&mut self, value: T) -> Result<(), E> {
            self.output.push(value);
            Ok(())
        }
    }
    let mut driver = Ordinary {
        items,
        test,
        output: Vec::new(),
    };
    run(&mut driver, invert)?;
    Ok(driver.output)
}
pub(crate) fn control_bytes<D: Driver>() -> Option<usize> {
    use std::mem::size_of;
    size_of::<D>()
        .checked_add(size_of::<D::Item>())?
        .checked_add(size_of::<Result<Option<D::Item>, D::Error>>())?
        .checked_add(size_of::<Result<bool, D::Error>>())?
        .checked_add(size_of::<Result<(), D::Error>>())?
        .checked_add(size_of::<(&mut D, bool)>())
}
