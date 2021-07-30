#![cfg_attr(feature = "allocator_api", feature(allocator_api))]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
//! This is a way to "transmute" Vecs soundly.
//! ```
//! # #[cfg(feature = "allocator_api")]
//! # {
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
//! # }
//! ```

extern crate alloc;

use alloc::vec::Vec;
use bytemuck::Pod;
#[cfg(feature = "allocator_api")]
use core::alloc::{AllocError, Allocator, Layout};
use core::{
    cmp::Ordering,
    fmt::Display,
    hint,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ptr::{self, NonNull},
    slice,
    sync::atomic::{self, AtomicPtr},
};

#[cfg(feature = "std")]
use std::error::Error;

/// Error for Vec transmutes.
/// It will always be `Alignment` -> `Length` -> `Capacity`
#[derive(Debug, PartialEq, Eq)]
pub enum TransmuteError {
    /// When the alignment of vec is incorrect.
    Alignment,
    /// When the length wouldn't be able to fit.
    Length,
    /// When the capacity wouldn't be able to fit.
    Capacity,
}

impl Display for TransmuteError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TransmuteError::Alignment => write!(formatter, "Alignment: alignment of the pointer to I must be equal or greater than the alignment of O"),
            TransmuteError::Length => write!(formatter, "Length: I's items cant fit in O's (e.g. 1 u8 to 1 u16)"),
            TransmuteError::Capacity => write!(formatter, "Capacity: capacicty of vec is incorrect"),
        }
    }
}

#[cfg(feature = "std")]
impl Error for TransmuteError {}

#[cfg(feature = "allocator_api")]
/// Handling correcting the alignment fed into the inner allocator.
#[derive(Debug)]
pub struct AlignmentCorrectorAllocator<I, O, A: Allocator> {
    allocator: A,
    ptr: AtomicPtr<O>,
    phantom: PhantomData<I>,
}
#[cfg(feature = "allocator_api")]
impl<I, O, A: Allocator> AlignmentCorrectorAllocator<I, O, A> {
    /// Create new `AlignmentCorrectorAllocator`.
    ///
    /// # Safety
    /// The `ptr` must be allocated with the `allocator` in the alignment of `I`.
    pub unsafe fn new(ptr: NonNull<O>, allocator: A) -> Self {
        Self {
            allocator,
            ptr: AtomicPtr::new(ptr.as_ptr()),
            phantom: PhantomData::default(),
        }
    }
    /// Create a new `AlignmentCorrectorAllocator` that acts the same as the allocator fed into it.
    pub fn new_null(allocator: A) -> Self {
        Self {
            allocator,
            ptr: AtomicPtr::new(ptr::null_mut()),
            phantom: PhantomData::default(),
        }
    }

    unsafe fn get_layout(&self, mut layout: Layout) -> Layout {
        let mut old = self.ptr.load(atomic::Ordering::Relaxed);
        if old.is_null() {
            return layout;
        }
        loop {
            match self.ptr.compare_exchange_weak(
                old,
                ptr::null_mut(),
                atomic::Ordering::SeqCst,
                atomic::Ordering::Relaxed,
            ) {
                Ok(x) if !x.is_null() => {
                    layout =
                        // SAFETY: the layout size must be correct for this I's alignment because it the
                        // creator of a AlignmentCorrectorAllocator ensures that.
                        // It is now a null ptr because correcting the allocation now would be invalid
                        // since the pointer has been recovered.
                        unsafe { Layout::from_size_align_unchecked(layout.size(), mem::align_of::<I>()) };

                    break layout;
                }
                Ok(_) => break layout,
                Err(x) if x.is_null() => break layout,
                Err(x) => old = x,
            }
        }
    }
}

#[cfg(feature = "allocator_api")]
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
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.allocator.deallocate(ptr, self.get_layout(layout)) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe {
            self.allocator
                .grow(ptr, self.get_layout(old_layout), new_layout)
        }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe {
            self.allocator
                .grow_zeroed(ptr, self.get_layout(old_layout), new_layout)
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: all conditions must be upheld by the caller
        unsafe { self.allocator.shrink(ptr, old_layout, new_layout) }
    }
}

#[cfg(feature = "allocator_api")]
unsafe fn from_raw_parts<I, O, A: Allocator>(
    old_ptr: NonNull<O>,
    length: usize,
    capacity: usize,
    allocator: A,
) -> Vec<O, AlignmentCorrectorAllocator<I, O, A>> {
    // SAFETY: the caller uploads the constrants for from_raw_parts except for the alignment of the
    // allocation that created the old pointer
    unsafe {
        Vec::<O, AlignmentCorrectorAllocator<I, O, A>>::from_raw_parts_in(
            old_ptr.as_ptr(),
            length,
            capacity,
            AlignmentCorrectorAllocator::<I, O, A>::new(old_ptr, allocator),
        )
    }
}

#[cfg(feature = "allocator_api")]
/// Whether or not a copy occured.
/// Also the copy variant doesn't have the custom allocator, and is therefore one usize smaller.
pub enum CopyNot<I, O, A: Allocator> {
    /// Copy occured.
    Copy(Vec<O, A>),
    /// There was no copy.
    Not(Vec<O, AlignmentCorrectorAllocator<I, O, A>>),
}

#[cfg(feature = "allocator_api")]
/// [`transmute_vec_may_copy`] but it tells you whether or not a copy occured and returns a normal
/// vec if it doesn't.
pub fn transmute_vec_copy_enum<I: Pod, O: Pod, A: Allocator>(input: Vec<I, A>) -> CopyNot<I, O, A> {
    match transmute_vec(input) {
        Ok(x) => CopyNot::Not(x),
        Err((old_vec, err)) => match err {
            TransmuteError::Alignment => {
                // I don't have to deal with ZSTs because transmute_vec will never fail with a ZST.
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
                for x in bytes_slice.chunks_exact(mem::size_of::<O>()) {
                    // SAFETY: O is Pod and the slice is the same length as the type.
                    // also its
                    return_vec.push(unsafe { ptr::read_unaligned(x.as_ptr().cast()) });
                }
                // freeing memory
                // SAFETY: this size and align come from a vec and the allocator is the same one
                // that allocated the memory. i dont have to call drop because its Pod
                unsafe {
                    let align = mem::align_of::<I>();
                    let size = mem::size_of::<I>() * capacity;
                    let layout = Layout::from_size_align_unchecked(size, align);
                    // ensuring i dont deallocate unallocated memory
                    if size != 0 {
                        return_vec
                            .allocator()
                            .deallocate(NonNull::new_unchecked(ptr.cast()), layout);
                    }
                };
                CopyNot::Copy(return_vec)
            }
            TransmuteError::Capacity | TransmuteError::Length => {
                // I don't have to deal with ZSTs because transmute_vec will never fail with a ZST.
                let (ptr, length, capacity, allocator) = {
                    let mut me = ManuallyDrop::new(old_vec);
                    (me.as_mut_ptr(), me.len(), me.capacity(), unsafe {
                        ptr::read(me.allocator())
                    })
                };

                // SAFETY: the divide rounds down so the length is correct
                let return_vec = unsafe {
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
                    let layout = Layout::from_size_align_unchecked(size, align);
                    // ensuring i dont deallocate ZSTs or unallocated memory
                    if size != 0 {
                        return_vec
                            .allocator()
                            .deallocate(NonNull::new_unchecked(ptr.cast()), layout);
                    }
                };
                CopyNot::Copy(return_vec)
            }
        },
    }
}

#[cfg(feature = "allocator_api")]
/// Same as `transmute_vec` but in case of an error it copies instead.
/// If it's over the length it removes whatever doesn't fit.
///
/// You may want to use [`transmute_vec_copy_enum`].
pub fn transmute_vec_may_copy<I: Pod, O: Pod, A: Allocator>(
    input: Vec<I, A>,
) -> Vec<O, AlignmentCorrectorAllocator<I, O, A>> {
    match transmute_vec_copy_enum(input) {
        CopyNot::Copy(x) => {
            let (ptr, length, capacity, allocator) = {
                let mut me = ManuallyDrop::new(x);
                (me.as_mut_ptr(), me.len(), me.capacity(), unsafe {
                    ptr::read(me.allocator())
                })
            };

            // SAFETY: comes directly from vec and AlignmentCorrectorAllocator::new_null
            // doesn't interfere with the allocator within
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
/// ```should_panic
/// # #![feature(allocator_api)]
/// # use transvec::transmute_vec;
/// let input: Vec<u8> = Vec::with_capacity(3);
/// let output: Vec<u16, _> = transmute_vec(input).unwrap();
/// ```
/// 3. The alignment of the vec is wrong.
///
/// Alignment, then length, then capacity will always be returned.
/// # ZSTs
/// 1. Anything -> ZST
///     - Keeps length, deallocates data.
/// 2. ZST -> Non ZST
///     - New Vec from previous allocator.
/// 3. Just don't do this.
///
/// # See also
/// - [`transmute_vec_may_copy`] -- Infailable
/// - [`transmute_vec_basic`] -- Returns a Vec without a specical allocator and works on stable, but only works with types with the same alignment.
#[cfg(feature = "allocator_api")]
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
    if mem::size_of::<O>() == 0 {
        // freeing memory
        // SAFETY: this size and align come from a vec and the allocator is the same one
        // that allocated the memory. i dont have to call drop because its Pod
        unsafe {
            let align = mem::align_of::<I>();
            let size = mem::size_of::<I>() * capacity;
            // ensuring i dont deallocate ZSTs or stuff with 0 capacity.
            if size != 0 {
                let layout = Layout::from_size_align_unchecked(size, align);
                allocator.deallocate(NonNull::new_unchecked(ptr.cast()), layout);
            }
        };

        let mut return_vec =
            Vec::with_capacity_in(capacity, AlignmentCorrectorAllocator::new_null(allocator));
        unsafe { return_vec.set_len(length) };

        return Ok(return_vec);
    } else if mem::size_of::<I>() == 0 || capacity == 0 {
        // freeing memory
        // SAFETY: this size and align come from a vec and the allocator is the same one
        // that allocated the memory. i dont have to call drop because its Pod
        unsafe {
            let align = mem::align_of::<I>();
            let size = mem::size_of::<I>() * capacity;
            // ensuring i dont deallocate ZSTs or stuff with 0 capacity.
            if size != 0 {
                let layout = Layout::from_size_align_unchecked(size, align);
                allocator.deallocate(NonNull::new_unchecked(ptr.cast()), layout);
            }
        };
        return Ok(Vec::new_in(AlignmentCorrectorAllocator::new_null(
            allocator,
        )));
    }

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
                // vecs also only give out nonnull ptrs
                Ok(unsafe {
                    from_raw_parts(
                        NonNull::new_unchecked(ptr).cast(),
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
                Ok(unsafe {
                    from_raw_parts(
                        NonNull::new_unchecked(ptr).cast(),
                        length,
                        capacity,
                        allocator,
                    )
                })
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

/// If alignment is the same this function is preferred over [`transmute_vec`].
/// # Errors
/// 1. The length of the vector wouldn't fit the type.
/// ```should_panic
/// # #![feature(allocator_api)]
/// # use transvec::transmute_vec_basic;
/// let input: Vec<u8> = vec![1, 2, 3];
/// let output: Vec<u16, _> = transmute_vec_basic(input).unwrap();
/// ```
/// 2. The capacity can't be converted to units of the output type.
/// ```should_panic
/// # #![feature(allocator_api)]
/// # use transvec::transmute_vec_basic;
/// let input: Vec<u8> = Vec::with_capacity(3);
/// let output: Vec<u16, _> = transmute_vec_basic(input).unwrap();
/// ```
/// Length, then capacity will be returned.
/// # ZSTs
/// 1. Anything -> ZST
///     - Keeps length, deallocates data.
/// 2. ZST -> Non ZST
///     - New Vec from previous allocator.
/// 3. Just don't do this.
/// 
/// # Panics
/// Panics if the alignment is not the same.
/// (This may be turned into a compile time error in the future, when possible without using nightly
/// features).
///
/// Otherwise this acts exactly the same as [`transmute_vec`].
pub fn transmute_vec_basic<I: Pod, O: Pod>(
    input: Vec<I>,
) -> Result<Vec<O>, (Vec<I>, TransmuteError)> {
    let (ptr, length, capacity) = {
        let mut me = ManuallyDrop::new(input);
        (me.as_mut_ptr(), me.len(), me.capacity())
    };

    if mem::align_of::<I>() != mem::align_of::<O>() {
        panic!(
            "Alignment of {} and {} are not the same",
            core::any::type_name::<I>(),
            core::any::type_name::<O>(),
        );
    } else if mem::size_of::<O>() == 0 {
        // SAFETY: this came directly from a vec
        drop(unsafe { Vec::from_raw_parts(ptr, length, capacity) });
        let mut vec = Vec::with_capacity(capacity);
        // SAFETY: its a zst so this is allowed
        unsafe { vec.set_len(length) };
        return Ok(vec);
    } else if mem::size_of::<I>() == 0 || capacity == 0 {
        // SAFETY: this came directly from a vec
        drop(unsafe { Vec::from_raw_parts(ptr, length, capacity) });
        return Ok(Vec::new());
    }

    match mem::size_of::<I>().cmp(&mem::size_of::<O>()) {
        Ordering::Greater | Ordering::Less => {
            if (length * mem::size_of::<I>()) % mem::size_of::<O>() != 0 {
                Err((
                    // SAFETY: this came directly from a vec
                    unsafe { Vec::from_raw_parts(ptr, length, capacity) },
                    TransmuteError::Length,
                ))
            } else if (capacity * mem::size_of::<I>()) % mem::size_of::<O>() != 0 {
                Err((
                    // SAFETY: this came directly from a vec
                    unsafe { Vec::from_raw_parts(ptr, length, capacity) },
                    TransmuteError::Capacity,
                ))
            } else {
                // SAFETY: the length and capacity of vec is corrected to be the correct size,
                // and its not discarding bytes on the end. the alignment is also checked.
                Ok(unsafe {
                    Vec::from_raw_parts(
                        ptr.cast(),
                        (length * mem::size_of::<I>()) / mem::size_of::<O>(),
                        (capacity * mem::size_of::<I>()) / mem::size_of::<O>(),
                    )
                })
            }
        }
        Ordering::Equal => {
            // SAFETY: they are both the same size and Pod
            Ok(unsafe {
                Vec::from_raw_parts(
                    ptr.cast(),
                    (length * mem::size_of::<I>()) / mem::size_of::<O>(),
                    (capacity * mem::size_of::<I>()) / mem::size_of::<O>(),
                )
            })
        }
    }
}

/// [`transmute_vec_basic`] but on fail it copies instead.
#[must_use = "You shouldn't use this function if you're not going to use the result as it's useless otherwise"]
pub fn transmute_vec_basic_copy<I: Pod, O: Pod>(input: Vec<I>) -> Vec<O> {
    match transmute_vec_basic(input) {
        Ok(x) => x,
        Err((old_vec, err)) => match err {
            TransmuteError::Alignment => unsafe { hint::unreachable_unchecked() },
            TransmuteError::Capacity | TransmuteError::Length => {
                // I don't have to deal with ZSTs because transmute_vec will never fail with a ZST.
                let ptr = old_vec.as_ptr();
                let length = old_vec.len();
                // SAFETY: the divide rounds down so the length is correct
                unsafe {
                    slice::from_raw_parts(
                        ptr.cast::<O>(),
                        (length * mem::size_of::<I>()) / mem::size_of::<O>(),
                    )
                }
                .to_vec()
            }
        },
    }
}

#[cfg(test)]
mod tests {

    use alloc::{vec, vec::Vec};
    #[cfg(feature = "allocator_api")]
    use crate::{transmute_vec, transmute_vec_may_copy};
    use crate::{
        transmute_vec_basic, transmute_vec_basic_copy,
        TransmuteError,
    };

    #[test]
    #[cfg(feature = "allocator_api")]
    // It does work with the same sized types
    fn basic_functioning() {
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 6];
        let output: Vec<i8, _> = match transmute_vec(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        assert_eq!(&output, &[0, 1, 2, 3, 4, 6]);
    }
    #[test]
    // It does work with the same sized types

    fn basic_basic_functioning() {
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 6];
        let output: Vec<i8> = match transmute_vec_basic(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        assert_eq!(&output, &[0, 1, 2, 3, 4, 6]);
    }

    #[test]
    #[should_panic]
    fn unalign_panic() {
        let _ = transmute_vec_basic::<u8, u16>(Vec::new());
    }

    #[test]
    fn different_size_same_align() {
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 6];
        let output: Vec<[u8; 2]> = match transmute_vec_basic(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        assert_eq!(&output, &[[0, 1], [2, 3], [4, 6]])
    }

    #[test]
    #[cfg(feature = "allocator_api")]
    fn small_to_large() {
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 6];
        let output: Vec<u16, _> = match transmute_vec(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        if cfg!(target_endian = "big") {
            assert_eq!(&output, &[1, 515, 1030]);
        } else {
            assert_eq!(&output, &[256, 770, 1540]);
        }
    }
    #[test]
    #[cfg(feature = "allocator_api")]
    fn large_to_small() {
        let input: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let output: Vec<u8, _> = match transmute_vec(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        if cfg!(target_endian = "big") {
            assert_eq!(&output, &[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8]);
        } else {
            assert_eq!(&output, &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0]);
        }
    }

    #[test]
    fn to_zsts_basic() {
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        let output: Vec<()> = match transmute_vec_basic(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        assert_eq!(output.len(), 6);
    }

    #[test]
    #[cfg(feature = "allocator_api")]
    fn add_and_remove() {
        let input: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut output: Vec<u8, _> = match transmute_vec(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        output.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        for _ in 0..10 {
            output.pop();
        }
        output.shrink_to_fit()
    }

    #[test]
    #[cfg(feature = "allocator_api")]
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

    #[test]
    #[cfg(feature = "allocator_api")]
    fn wrong_length_copy() {
        let input: Vec<u8> = vec![1, 2, 3];
        let output: Vec<u16, _> = transmute_vec_may_copy::<_, u16, _>(input);
        if cfg!(target_endian = "big") {
            assert_eq!(&output, &[258]);
        } else {
            assert_eq!(&output, &[513]);
        }
    }

    #[test]
    #[cfg(feature = "allocator_api")]
    fn may_copy() {
        let input: Vec<u8> = vec![1, 2];
        let output: Vec<u16, _> = transmute_vec_may_copy::<_, u16, _>(input);
        if cfg!(target_endian = "big") {
            assert_eq!(&output, &[258]);
        } else {
            assert_eq!(&output, &[513]);
        }
    }

    #[test]
    fn basic_copy() {
        let input: Vec<u8> = vec![1, 2];
        let output: Vec<[u8; 2]> = transmute_vec_basic_copy(input);
        assert_eq!(&output, &[[1, 2]]);
    }

    #[test]
    #[cfg(feature = "allocator_api")]
    fn from_zsts() {
        let input = vec![(), (), ()];
        let output: Vec<u8, _> = match transmute_vec(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        assert_eq!(output.len(), 0);
    }

    #[test]
    #[cfg(feature = "allocator_api")]
    fn to_zsts() {
        let input: Vec<u8> = vec![0, 1, 2, 3, 4, 5];
        let output: Vec<(), _> = match transmute_vec(input) {
            Ok(x) => x,
            Err(_) => return,
        };
        assert_eq!(output.len(), 6);
    }
}
