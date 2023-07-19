use crate::{bitstreams::InputBitStream, webgraph::BVGraph, nat2int};

pub trait BVGraphSectionIterator: Iterator {
    fn skip(&mut self, n: usize) -> usize;
}

/// An iterator returning the residuals of a node.
pub struct ResidualIntIterator<'a> {
    /// The graph associated to this iterator.
    g: &'a BVGraph,
    /// The input bit stream from which residuals will be read.
    ibs: &'a mut InputBitStream,
    /// The last residual returned.
    next: i64,
    /// The number of remaining residuals.
    remaining: usize,
}

impl<'a> Iterator for ResidualIntIterator<'a> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let result = self.next;
        self.remaining -= 1;
        if self.remaining != 0 {
            self.next += self.g.read_residual(self.ibs).unwrap() as i64 + 1;
        }
        Some(result)
    }
}

impl<'a> ResidualIntIterator<'a> {
    fn new(g: &'a BVGraph, ibs: &'a mut InputBitStream, residual_count: usize, x: i64) -> Self {
        Self {
            g,
            remaining: residual_count,
            ibs,
            next: x + nat2int(g.read_residual(ibs).unwrap() as u64) as i64
        }
    }
}

impl<'a> BVGraphSectionIterator for ResidualIntIterator<'a> {
    fn skip(&mut self, n: usize) -> usize { // TODO: as implementation of Iterator
        if n >= self.remaining {
            let n = self.remaining;
            self.remaining = 0;
            return n;
        }
    
        let mut i = n;
        while i != 0{
            self.next += self.g.read_residual(self.ibs).unwrap() as i64 + 1;
            i -= 1;
        }
    
        self.remaining -= n;
        n
    }
}

/// An iterator returning the integers contained in a sequence of intervals.
pub struct IntIntervalSequenceIterator {
    /// The left extremes.
    left: Box<[usize]>,
    /// The lengths.
    len: Box<[usize]>,
    /// The number of remaining intervals (including the current one). It is zero exactly when the iterator is exhausted. 
    remaining: usize,
    /// The index of the current interval.
    curr_interval: usize,
    /// The current position in the current interval: the next integer to be output is `curr_left` + `curr_index`. 
    curr_index: usize,
    /// The left point of the current interval.
    curr_left: usize,
}

impl Iterator for IntIntervalSequenceIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let next = self.curr_left + self.curr_index;
        self.curr_index += 1;

        if self.curr_index == self.len[self.curr_interval] {
            self.advance();
        }

        Some(next)
    }
}

impl IntIntervalSequenceIterator {
    /// Creates a new interval-sequence iterator by specifying
    /// arrays of left extremes and lengths, and the number of valid entries.
    /// 
    /// # Arguments
    /// 
    /// * `left` - an array containing the left extremes of the intervals generating this iterator.
    /// * `len` - an array (of the same length as `left`) containing the number of integers (greater than zero) in each interval.
    /// * `n` - the number of valid entries in `left` and `len`.
    pub fn new(left: Box<[usize]>, len: Box<[usize]>, n: usize) -> Self {
        Self {
            left,
            len,
            remaining: n,
            curr_left: if n != 0 {left[0]} else {0},
            curr_index: 0,
            curr_interval: 0
        }
    }

    fn advance(&mut self) {
        self.remaining -= 1;

        if self.remaining != 0 {
            self.curr_interval += 1;
            self.curr_left = self.left[self.curr_interval];
        }

        self.curr_index = 0;
    }
}

impl BVGraphSectionIterator for IntIntervalSequenceIterator {
    fn skip(&mut self, n: usize) -> usize { // TODO: as implementation of Iterator
        let mut skipped = 0;
    
        while skipped < n && self.remaining != 0 {
            if n - skipped < self.len[self.curr_interval] - self.curr_index {
                self.curr_index += n - skipped;
                return n;
            } else {
                skipped += self.len[self.curr_interval] - self.curr_index;
                self.advance();
            }
        }
    
        skipped
    }
}

/// An iterator returning the union of the integers returned by two `Iterator`s.
/// The two iterators must return integers in an increasing fashion; the resulting
/// [`MergedIntIterator`] will do the same. Duplicates will be eliminated.
pub struct MergedIntIterator<'a> {
    /// The first component iterator.
    it0: &'a dyn Iterator<Item = usize>,
    /// The second component iterator.
    it1: &'a dyn Iterator<Item = usize>,
    /// The last integer returned by `it0`.
    curr0: Option<usize>,
    /// The last integer returned by `it1`.
    curr1: Option<usize>
}

impl Iterator for MergedIntIterator<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr0.is_none() {
            let result = self.curr1;
            self.curr1 = self.it1.next();
            return result;
        }
        if self.curr1.is_none() {
            let result = self.curr0;
            self.curr0 = self.it0.next();
            return result;
        }
        if self.curr0.unwrap() < self.curr1.unwrap() {
            let result = self.curr0;
            self.curr0 = self.it0.next();
            return result;
        } else {
            let result = self.curr1;
            if self.curr0.unwrap() == self.curr1.unwrap() {
                self.curr0 = self.it0.next();
            }
            self.curr1 = self.it1.next();
            return result;
        }
    }
}

impl<'a> MergedIntIterator<'a> {
    /// Creates a new merged iterator by mergin two given iterators; the resulting iterator will not emit more than `n` integers.
    /// 
    /// # Arguments
    /// * `it0` - the first (monotonically nondecreasing) component iterator.
    /// * `it1` - the second (monotonically nondecreasing) component iterator.
    pub fn new(it0: &'a dyn Iterator<Item = usize>, it1: &'a dyn Iterator<Item = usize>) -> Self {
        Self {
            it0,
            it1,
            curr0: it0.next(),
            curr1: it1.next()
        }
    }
}

impl<'a> BVGraphSectionIterator for MergedIntIterator<'a> {
    fn skip(&mut self, n: usize) -> usize { // TODO: as implementation of Iterator
        let mut i: usize;
        while i < n {
            match (self.curr0, self.curr1) {
                (None, None) => break,
                (None, Some(Curr1)) => self.curr1 = self.it1.next(),
                (Some(curr0), Some(curr1)) if curr0 < curr1 => self.curr0 = self.it0.next(),
                (Some(curr0), Some(curr1)) if curr0 > curr1 => self.curr1 = self.it1.next(),
                (Some(curr0), Some(curr1)) /* if curr0 == curr1 */ => {self.curr0 = self.it0.next(); self.curr1 = self.it1.next()}
                (Some(curr0), None) => self.curr0 = self.it0.next()
            };

            i += 1;
        }

        i
    }
}

/// An iterator returning the element of an underlying iterator while
/// filtering them using an inclusion-exclusion block list.
/// 
/// A *mask* is an array of integers. The sum of the values contained in the mask
/// must not exceed the number of elements returned by the underlying iterator. Moreover, all integers in the mask
/// must be positive, except possibly for the first one, which may be zero.
/// 
/// Mask values are interpreted as specifying inclusion-exclusion blocks.
/// Suppose that the underlying iterator returns *N* values, and that the mask is
/// *n*<sub>0</sub>, *n*<sub>1</sub>, ..., *n*<sub>k</sub>.
/// Then, the first *n*<sub>0</sub> values returned by the underlying iterator must be kept,
/// the next *n*<sub>1</sub> values must be ignored, the next *n*<sub>2</sub> must be kept
/// and so on. The last *N*-(*n*<sub>0</sub> + ... + *n*<sub>k</sub>)
/// must be kept if *k* is odd, and must be ignored otherwise.
/// An instance of this class will return the kept values only, in increasing order.
pub struct MaskedIntIterator {
    /// The underlying iterator.
    underlying: Box<dyn BVGraphSectionIterator<Item = usize>>,
    /// The mask
    mask: Box<[usize]>,
    /// The mask's length.
    mask_len: usize,
    /// This index in the mask always represents an exclusion block.
    curr_mask: usize,
    /// How many integers are left in the current inclusion block. If `0` everything left must be discarded;
    /// if `None` all remaining values must be kept.
    left: Option<usize>,
}

impl Iterator for MaskedIntIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left.is_some() && self.left.unwrap() == 0 {
            return None;
        }

        let next = self.underlying.next();

        if self.left.is_none() || next.is_none() {
            return next;
        }

        if self.left.unwrap() > 0 {
            self.left = Some(self.left.unwrap() - 1);
            self.advance();
        }

        next
    }
}

impl MaskedIntIterator {
    /// Creates a new masked iterator using a given mask, mask length, and underlying iterator.
    /// 
    /// # Arguments
    /// 
    /// * `mask` - the mask.
    /// * `mask_len` - the explicit mask length.
    /// * `underlying` - the underlying iterator.
    pub fn new(mask: Box<[usize]>, mask_len: usize, underlying: Box<dyn BVGraphSectionIterator<Item = usize>>) -> Self {
        let left;
        let mut curr_mask = 0;
        if mask_len != 0 {
            left = Some(mask[curr_mask]);
            curr_mask += 1;
        } else {
            left = None;
        }

        let mut obj = Self {mask, mask_len, curr_mask, left, underlying};

        if mask_len != 0 {
            obj.advance();
        }

        obj
    }

    fn advance(&mut self) {
        assert!(self.left.is_some());

        if self.left.unwrap() == 0 && self.curr_mask < self.mask_len {
            BVGraphSectionIterator::skip(&mut *self.underlying, self.mask[self.curr_mask]);
            self.curr_mask += 1;
            self.left = if self.curr_mask < self.mask_len {self.curr_mask += 1; Some(self.mask[self.curr_mask - 1])} else {None};
        }
    }
}

impl BVGraphSectionIterator for MaskedIntIterator {
    fn skip(&mut self, n: usize) -> usize { // TODO: as implementation of Iterator
        let mut skipped = 0;

        while skipped < n && (self.left.is_none() || self.left.unwrap() != 0) {
            if self.left.is_none() {
                let result = BVGraphSectionIterator::skip(&mut *self.underlying, n);
                skipped += result;
                if skipped < n {
                    break;
                }
            } else if n - skipped < self.left.unwrap() {
                BVGraphSectionIterator::skip(&mut *self.underlying, n - skipped);
                self.left = Some(self.left.unwrap() - n - skipped);
                return n;
            } else {
                BVGraphSectionIterator::skip(&mut *self.underlying, self.left.unwrap());
                skipped += self.left.unwrap();
                self.left = Some(0);
                self.advance();
            }
        }

        skipped
    }
}