#![feature(allocator_api)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

#![warn(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![allow(incomplete_features)]
//! This is a way to "transmute" Vecs soundly.
//! ```
//! #![feature(allocator_api)] // this requires the allocator api because the way that this
//! // handles deallocating hooks into the allocator api
//! use transvec::transmute_vec;
//! let input_vec: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
//! let output: Vec<u8, _> = match transmute_vec(input_vec) {
//!     Ok(x) => x,
//!     // the "transmute" can fail, if the alignment/capacity/length is incorrect
//!     // consider using `transmute_vec_may_copy`
//!     Err((old_vec, err)) => return println!("Error: {:?}", err),
//! };
//! if cfg!(target_endian = "big") {
//!     assert_eq!(
//!         &output,
//!         &[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]
//!     );
//! } else {
//!     assert_eq!(
//!         &output,
//!         &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]
//!     );
//! }

use bytemuck::Pod;
use chunk_iter::ChunkIter;
use core::slice;
use std::{
    alloc::{AllocError, Allocator, Layout},
    cmp::Ordering,
    fmt::Display,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ptr::{self, NonNull},
};
use thiserror::Error;

/// Error for Vec transmutes.
/// It will always be `Alignment` -> `Length` -> `Capacity`
#[derive(Error, Debug, PartialEq, Eq)]
pub enum TransmuteError {
    #[error("alignment of vec is incorrect")]
    /// When the alignment of vec is incorrect.
    Alignment,
    #[error("length of vec is incorrect")]
    /// When the length wouldn't be able to fit.
    Length,
    #[error("capacicty of vec is incorrect")]
    /// When the capacity wouldn't be able to fit.
    Capacity,
}

/// When the length doesn't line up correctly in `[transmute_vec_may_copy]`.
#[derive(Error, Debug, PartialEq, Eq)]
pub struct Length;

impl Display for Length {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Length is incorrect.")
    }
}

/// Implementation detail: Do not use
#[derive(Copy, Clone, Debug)]
pub struct AlignmentCorrectorAllocator<I, O, A: Allocator> {
    allocator: A,
    ptr: *const O,
    phantom: PhantomData<I>,
}
impl<I, O, A: Allocator> AlignmentCorrectorAllocator<I, O, A> {
    fn new(ptr: *const O, allocator: A) -> Self {
        Self {
            allocator,
            ptr,
            phantom: PhantomData::default(),
        }
    }
    fn new_null(allocator: A) -> Self {
        Self {
            allocator,
            ptr: ptr::null(),
            phantom: PhantomData::default(),
        }
    }
}
unsafe impl<I, O, A: Allocator> Allocator for AlignmentCorrectorAllocator<I, O, A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocator.allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocator.allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, mut layout: Layout) {
        if ptr.as_ptr() == self.ptr as *mut _ {
            layout =
                unsafe { Layout::from_size_align_unchecked(layout.size(), mem::align_of::<I>()) };
        }
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.allocator.deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        mut old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if ptr.as_ptr() == self.ptr as *mut _ {
            old_layout = unsafe {
                Layout::from_size_align_unchecked(old_layout.size(), mem::align_of::<I>())
            };
        };
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.allocator.grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        mut old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if ptr.as_ptr() == self.ptr as *mut _ {
            old_layout = unsafe {
                Layout::from_size_align_unchecked(old_layout.size(), mem::align_of::<I>())
            };
        };
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.allocator.grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        mut old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if ptr.as_ptr() == self.ptr as *mut _ {
            old_layout = unsafe {
                Layout::from_size_align_unchecked(old_layout.size(), mem::align_of::<I>())
            };
        };
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.allocator.shrink(ptr, old_layout, new_layout) }
    }
}

unsafe fn from_raw_parts<I, O, A: Allocator>(
    old_ptr: *mut O,
    length: usize,
    capacity: usize,
    allocator: A,
) -> Vec<O, AlignmentCorrectorAllocator<I, O, A>> {
    // SAFETY: the caller uploads the constrants for from_raw_parts except for the alignment of the
    // allocation that created the old pointer
    unsafe {
        Vec::<O, AlignmentCorrectorAllocator<I, O, A>>::from_raw_parts_in(
            old_ptr,
            length,
            capacity,
            AlignmentCorrectorAllocator::<I, O, A>::new(old_ptr, allocator),
        )
    }
}

/// Whether or not a copy occured.
/// Also the copy variant doesn't have the custom allocator, and is therefore one usize smaller.
pub enum CopyNot<I, O, A: Allocator> {
    /// Copy occured
    Copy(Vec<O, A>),
    /// There was no copy.
    Not(Vec<O, AlignmentCorrectorAllocator<I, O, A>>),
}

/// [`transmute_vec_may_copy`] but it tells you whether or not a copy occured and returns a normal
/// vec if it doesn't.
pub fn transmute_vec_copy_enum<I: Pod, O: Pod, A: Allocator>(input: Vec<I, A>) -> CopyNot<I, O, A>
where
    [(); mem::size_of::<O>()]: , // todo: see if i can remove this where
{
    match transmute_vec(input) {
        Ok(x) => CopyNot::Not(x),
        Err((old_vec, err)) => match err {
            TransmuteError::Alignment => {
                let (ptr, length, capacity, allocator) = {
                    let mut me = ManuallyDrop::new(old_vec);
                    (me.as_mut_ptr(), me.len(), me.capacity(), unsafe {
                        ptr::read(me.allocator())
                    })
                };
                

                // SAFETY: the ptr comes from a vec and the length is calcuated properly
                let bytes_slice = unsafe {
                    slice::from_raw_parts(ptr.cast::<u8>(), length * mem::size_of::<I>())
                };
                let mut return_vec = Vec::with_capacity_in(
                    (length * mem::size_of::<I>()) / mem::size_of::<O>(),
                    allocator,
                );
                for x in bytes_slice
                    .iter()
                    .copied()
                    .chunks::<{ mem::size_of::<O>() }>()
                {
                    // SAFETY: O is Pod and the array is the same length as the type.
                    return_vec.push(unsafe { mem::transmute_copy(&x) })
                }
                // freeing memory
                // SAFETY: this size and align come from a vec and the allocator is the same one
                // that allocated the memory. i dont have to call drop because its Pod
                unsafe { 
                    let align = mem::align_of::<I>();
                    let size = mem::size_of::<I>() * capacity;
                    let layout = Layout::from_size_align_unchecked(size, align) ;
                    return_vec.allocator().deallocate(NonNull::new_unchecked(ptr.cast()), layout);
                };
                CopyNot::Copy(return_vec)
            }
            TransmuteError::Capacity | TransmuteError::Length => {
                let (ptr, length, capacity, allocator) = {
                    let mut me = ManuallyDrop::new(old_vec);
                    (me.as_mut_ptr(), me.len(), me.capacity(), unsafe {
                        ptr::read(me.allocator())
                    })
                };

                // SAFETY: the divide rounds down so the length is correct
                let return_vec = 
                    unsafe {
                        slice::from_raw_parts(
                            ptr.cast::<O>(),
                            (length * mem::size_of::<I>()) / mem::size_of::<O>(),
                        )
                    }
                    .to_vec_in(allocator);
                
                // freeing memory
                // SAFETY: this size and align come from a vec and the allocator is the same one
                // that allocated the memory. i dont have to call drop because its Pod
                unsafe { 
                    let align = mem::align_of::<I>();
                    let size = mem::size_of::<I>() * capacity;
                    let layout = Layout::from_size_align_unchecked(size, align) ;
                    return_vec.allocator().deallocate(NonNull::new_unchecked(ptr.cast()), layout);
                };
                CopyNot::Copy(return_vec)

            }
        },
    }
}

/// Same as `transmute_vec` but in case of an error it copies instead.
/// If it's over the length it removes whatever doesn't fit.
/// You may want to use [`transmute_vec_copy_enum`].
pub fn transmute_vec_may_copy<I: Pod, O: Pod, A: Allocator>(
    input: Vec<I, A>,
) -> Vec<O, AlignmentCorrectorAllocator<I, O, A>>
where
    [(); mem::size_of::<O>()]: ,
{
    match transmute_vec_copy_enum(input) {
        CopyNot::Copy(x) => {
            let (ptr, length, capacity, allocator) = {
                let mut me = ManuallyDrop::new(x);
                (me.as_mut_ptr(), me.len(), me.capacity(), unsafe {
                    ptr::read(me.allocator())
                })
            };

            // SAFETY: comes directly from vec and AlignmentCorrectorAllocator::new_null
            // doesn't actually do anything
            unsafe {
                Vec::from_raw_parts_in(
                    ptr,
                    length,
                    capacity,
                    AlignmentCorrectorAllocator::new_null(allocator),
                )
            }
        }
        CopyNot::Not(x) => x,
    }
}

/// Allows transmuting of a Vec to another vec of a different size, with 0 copies.
/// # Example
/// ```
/// #![feature(allocator_api)] // this requires the allocator api because the way that this
/// // handles deallocating hooks into the allocator api
/// use transvec::transmute_vec;
/// let input_vec: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let output: Vec<u8, _> = match transmute_vec(input_vec) {
///     Ok(x) => x,
///     // the "transmute" can fail, if the alignment/capacity/length is incorrect
///     // consider using `transmute_vec_may_copy`
///     Err((old_vec, err)) => return println!("Error: {:?}", err),
/// };
/// if cfg!(target_endian = "big") {
///     assert_eq!(
///         &output,
///         &[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]
///     );
/// } else {
///     assert_eq!(
///         &output,
///         &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]
///     );
/// }
/// ```
/// # Errors
/// This errors when:
/// 1. The length of the vector wouldn't fit the type.
/// ```should_panic
/// # #![feature(allocator_api)]
/// # use transvec::transmute_vec;
/// let input: Vec<u8> = vec![1, 2, 3];
/// let output: Vec<u16, _> = transmute_vec(input).unwrap();
/// ```
/// 2. The capacity can't be converted to units of the output type.
/// 3. The alignment of the vec is wrong.
///
/// Alignment, then length, then capacity will always be returned.
/// # See also
/// - [`transmute_vec_may_copy`]
#[allow(clippy::type_complexity)]
pub fn transmute_vec<I: Pod, O: Pod, A: Allocator>(
    input: Vec<I, A>,
) -> Result<Vec<O, AlignmentCorrectorAllocator<I, O, A>>, (Vec<I, A>, TransmuteError)> {
    let (ptr, length, capacity, allocator) = {
        let mut me = ManuallyDrop::new(input);
        (me.as_mut_ptr(), me.len(), me.capacity(), unsafe {
            ptr::read(me.allocator())
        })
    };

    match mem::size_of::<I>().cmp(&mem::size_of::<O>()) {
        Ordering::Greater | Ordering::Less => {
            if ptr.align_offset(mem::align_of::<O>()) != 0 {
                Err((
                    // SAFETY: this came directly from a vec
                    unsafe { Vec::from_raw_parts_in(ptr, length, capacity, allocator) },
                    TransmuteError::Alignment,
                ))
            } else if (length * mem::size_of::<I>()) % mem::size_of::<O>() != 0 {
                Err((
                    // SAFETY: this came directly from a vec
                    unsafe { Vec::from_raw_parts_in(ptr, length, capacity, allocator) },
                    TransmuteError::Length,
                ))
            } else if (capacity * mem::size_of::<I>()) % mem::size_of::<O>() != 0 {
                Err((
                    // SAFETY: this came directly from a vec
                    unsafe { Vec::from_raw_parts_in(ptr, length, capacity, allocator) },
                    TransmuteError::Capacity,
                ))
            } else {
                // SAFETY: the length and capacity of vec is corrected to be the correct size,
                // and its not discarding bytes on the end. the alignment is also checked and on
                // drop, the custom allocator ensures deallocation is handled properly
                Ok(unsafe {
                    from_raw_parts(
                        ptr.cast(),
                        (length * mem::size_of::<I>()) / mem::size_of::<O>(),
                        (capacity * mem::size_of::<I>()) / mem::size_of::<O>(),
                        allocator,
                    )
                })
            }
        }
        Ordering::Equal => {
            if ptr.align_offset(mem::align_of::<O>()) == 0 {
                // SAFETY: its aligned and thats all that matters
                Ok(unsafe { from_raw_parts(ptr.cast(), length, capacity, allocator) })
            } else {
                Err((
                    // SAFETY: this came directly from a vec
                    unsafe { Vec::from_raw_parts_in(ptr, length, capacity, allocator) },
                    TransmuteError::Alignment,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::{transmute_vec, TransmuteError};

    #[test]
    // It does work with the same sized types
    fn basic_functioning() {
        let input_vec: Vec<u8> = vec![0, 1, 2, 3, 4, 6];
        let output: Vec<i8, _> = match transmute_vec(input_vec) {
            Ok(x) => x,
            Err((_, err)) => return println!("Error: {:?}", err),
        };
        assert_eq!(&output, &[0, 1, 2, 3, 4, 6]);
    }

    #[test]
    fn small_to_large() {
        let input_vec: Vec<u8> = vec![0, 1, 2, 3, 4, 6];
        let output: Vec<u16, _> = match transmute_vec(input_vec) {
            Ok(x) => x,
            Err((_, err)) => return println!("Error: {:?}", err),
        };
        if cfg!(target_endian = "big") {
            assert_eq!(&output, &[1, 515, 1030]);
        } else {
            assert_eq!(&output, &[256, 770, 1540]);
        }
    }
    #[test]
    fn large_to_small() {
        let input_vec: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let output: Vec<u8, _> = match transmute_vec(input_vec) {
            Ok(x) => x,
            Err((_, err)) => return println!("Error: {:?}", err),
        };
        if cfg!(target_endian = "big") {
            assert_eq!(&output, &[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]);
        } else {
            assert_eq!(&output, &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]);
        }
    }

    #[test]
    fn add_and_remove() {
        let input_vec: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut output: Vec<u8, _> = match transmute_vec(input_vec) {
            Ok(x) => x,
            Err((_, err)) => return println!("Error: {:?}", err),
        };
        output.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        for _ in 0..10 {
            output.pop();
        }
    }

    #[test]
    fn wrong_length() {
        let input: Vec<u8> = vec![1, 2, 3];
        match transmute_vec::<_, u16, _>(input) {
            Ok(_) => panic!(),
            Err((_, err)) => match err {
                TransmuteError::Alignment | TransmuteError::Length => (),
                x => panic!("{:?}", x),
            },
        };
    }
}
