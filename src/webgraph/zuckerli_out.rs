use std::{fs, vec, cmp::Ordering, marker::PhantomData, cell::{RefCell, Cell}, rc::Rc, borrow::BorrowMut, time::Instant, collections::HashMap};

use sucds::{mii_sequences::{EliasFanoBuilder, EliasFano}, Serializable};

use crate::{ImmutableGraph, properties::Properties, utils::{encodings::{UniversalCode, GammaCode, ZetaCode, Huffman, zuck_encode, K_ZUCK, I_ZUCK, J_ZUCK}, nat2int, int2nat}, huffman_zuckerli::{huffman_encoder::HuffmanEncoder, K_NUM_SYMBOLS}};
use crate::bitstreams::{BinaryReader, BinaryWriter};

pub const FIRST_DEGREE_CTX: usize = 0;
pub const DEGREE_BASE_CTX: usize = 1;
pub const NUM_DEGREE_CTX: usize = 32;
pub const REFERENCE_BASE_CTX: usize = DEGREE_BASE_CTX + NUM_DEGREE_CTX;
pub const NUM_REFERENCE_CTX: usize = 64;
pub const BLOCK_COUNT_CTX: usize = REFERENCE_BASE_CTX + NUM_REFERENCE_CTX; 
pub const BLOCK_CTX: usize = BLOCK_COUNT_CTX + 1;
pub const BLOCK_CTX_EVEN: usize = BLOCK_CTX + 1;
pub const BLOCK_CTX_ODD: usize = BLOCK_CTX_EVEN + 1;
pub const FIRST_RESIDUAL_BASE_CTX: usize = BLOCK_CTX_ODD + 1;
pub const NUM_FIRST_RESIDUAL_CTX: usize = 32;
pub const RESIDUALS_BASE_CTX: usize = FIRST_RESIDUAL_BASE_CTX + NUM_FIRST_RESIDUAL_CTX;
pub const NUM_RESIDUAL_CTX: usize = 80;
pub const RLE_CTX: usize = RESIDUALS_BASE_CTX + NUM_RESIDUAL_CTX;

pub const NUM_CONTEXTS: usize = RLE_CTX + 1;

#[derive(Clone, Eq, PartialEq, Debug, Default)]
struct CompressionVectors {
    blocks: RefCell<Vec<usize>>,
    extras: RefCell<Vec<usize>>,
    left: RefCell<Vec<usize>>,
    len: RefCell<Vec<usize>>,
    residuals: RefCell<Vec<usize>>,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct BVGraph<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> {
    n: usize,
    m: usize,
    pub graph_memory: Rc<[u8]>,
    pub offsets: Box<[usize]>,
    graph_binary_wrapper: RefCell<BinaryReader>,
    outdegrees_binary_wrapper: RefCell<BinaryReader>,
    cached_node: Cell<Option<usize>>,
    cached_outdegree: Cell<Option<usize>>,
    cached_ptr: Cell<Option<usize>>,
    in_max_ref_count: usize,
    in_window_size: usize,
    in_min_interval_len: usize,
    out_max_ref_count: usize,
    out_window_size: usize,
    out_min_interval_len: usize,
    in_zeta_k: Option<u64>,
    out_zeta_k: Option<u64>,
    compression_vectors: CompressionVectors,
    _phantom_in_block_coding: PhantomData<InBlockCoding>,
    _phantom_in_block_count_coding: PhantomData<InBlockCountCoding>,
    _phantom_in_outdegree_coding: PhantomData<InOutdegreeCoding>,
    _phantom_in_offset_coding: PhantomData<InOffsetCoding>,
    _phantom_in_reference_coding: PhantomData<InReferenceCoding>,
    _phantom_in_interval_coding: PhantomData<InIntervalCoding>,
    _phantom_in_residual_coding: PhantomData<InResidualCoding>,
    _phantom_out_block_coding: PhantomData<OutBlockCoding>,
    _phantom_out_block_count_coding: PhantomData<OutBlockCountCoding>,
    _phantom_out_outdegree_coding: PhantomData<OutOutdegreeCoding>,
    _phantom_out_offset_coding: PhantomData<OutOffsetCoding>,
    _phantom_out_reference_coding: PhantomData<OutReferenceCoding>,
    _phantom_out_interval_coding: PhantomData<OutIntervalCoding>,
    _phantom_out_residual_coding: PhantomData<OutResidualCoding>,
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> ImmutableGraph for BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
>
{
    type NodeT = usize;

    /// Returns the number of nodes in the BVGraph.
    #[inline]
    fn num_nodes(&self) -> usize {
        self.n
    }

    /// Returns the number of edges in the BVGraph.
    #[inline]
    fn num_arcs(&self) -> usize {
        self.m
    }

    /// Returns the outdegree of a given node or `None` otherwise.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number
    fn outdegree(&self, x: Self::NodeT) -> Option<usize> {
        if self.cached_node.get().is_some() && x == self.cached_node.get().unwrap() {
            return self.cached_outdegree.get();
        }
        
        if x >= self.n {
            return None;
        }

        self.outdegrees_binary_wrapper.borrow_mut().position(self.offsets[x] as u64);
        
        self.cached_node.set(Some(x));
        self.cached_outdegree.set(Some(InOutdegreeCoding::read_next(&mut self.outdegrees_binary_wrapper.borrow_mut(), self.in_zeta_k) as usize));
        self.cached_ptr.set(Some(self.outdegrees_binary_wrapper.borrow_mut().get_position()));

        self.cached_outdegree.get()
    }

    /// Returns the list of successors of a given node.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number
    fn successors(&self, x: usize) -> Box<[Self::NodeT]> {
        assert!(x < self.n, "Node index out of range {}", x);
        self.decode_list(x, &mut self.graph_binary_wrapper.borrow_mut(), None, &mut []).into_boxed_slice()
    }

    fn store(&mut self, basename: &str) -> std::io::Result<()> {
        let mut graph_obs = BinaryWriter::new();
        let mut offsets_obs = BinaryWriter::new();

        self.compress(&mut graph_obs, &mut offsets_obs);
        
        let graph = graph_obs.build();
        let offsets = offsets_obs.build();
        let props = Properties {
            nodes: self.n,
            arcs: self.m,
            window_size: self.out_window_size,
            max_ref_count: self.out_max_ref_count,
            min_interval_len: self.out_min_interval_len,
            zeta_k: self.out_zeta_k,
            outdegree_coding: OutOutdegreeCoding::to_encoding_type(),
            block_coding: OutBlockCoding::to_encoding_type(),
            residual_coding: OutResidualCoding::to_encoding_type(),
            interval_coding: OutIntervalCoding::to_encoding_type(),
            reference_coding: OutReferenceCoding::to_encoding_type(),
            block_count_coding: OutBlockCountCoding::to_encoding_type(),
            offset_coding: OutOffsetCoding::to_encoding_type()
        };

        fs::write(format!("{}.graph", basename), graph.os).unwrap();
        fs::write(format!("{}.offsets", basename), offsets.os).unwrap();
        fs::write(format!("{}.properties", basename), Into::<String>::into(props))?;

        Ok(())
    }
}

pub struct BVGraphNodeIterator<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
    BV: AsRef<BVGraph<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
>>>
{
    // The number of nodes
    n: usize,
    // The graph on which we iterate
    graph: BV,
    // The input bit stream
    pub ibs: BinaryReader,
    // The size of the cyclic buffer
    cyclic_buffer_size: usize,
    // Window to be passed to [`decode_list`]
    window: Vec<Vec<usize>>,
    // Outdegrees of the window's lists to be passed to [`decode_list`]
    outd: Vec<usize>,
    // The index of the node from which we started iterating
    from: usize,
    // The index of the node just before the next one
    curr: i64,
    _phantom_in_block_coding: PhantomData<InBlockCoding>,
    _phantom_in_block_count_coding: PhantomData<InBlockCountCoding>,
    _phantom_in_outdegree_coding: PhantomData<InOutdegreeCoding>,
    _phantom_in_offset_coding: PhantomData<InOffsetCoding>,
    _phantom_in_reference_coding: PhantomData<InReferenceCoding>,
    _phantom_in_interval_coding: PhantomData<InIntervalCoding>,
    _phantom_in_residual_coding: PhantomData<InResidualCoding>,
    _phantom_out_block_coding: PhantomData<OutBlockCoding>,
    _phantom_out_block_count_coding: PhantomData<OutBlockCountCoding>,
    _phantom_out_outdegree_coding: PhantomData<OutOutdegreeCoding>,
    _phantom_out_offset_coding: PhantomData<OutOffsetCoding>,
    _phantom_out_reference_coding: PhantomData<OutReferenceCoding>,
    _phantom_out_interval_coding: PhantomData<OutIntervalCoding>,
    _phantom_out_residual_coding: PhantomData<OutResidualCoding>,
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
    BV: AsRef<BVGraph<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
>>> Iterator for BVGraphNodeIterator<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
        BV
> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() {
            return None;
        }

        self.curr += 1;
        let curr_idx = self.curr as usize % self.cyclic_buffer_size;
        let decoded_list = self.graph.as_ref().decode_list(self.curr as usize, &mut self.ibs, Some(&mut self.window), &mut self.outd);

        let d = self.outd[curr_idx];

        if self.window[curr_idx].len() < d {
            self.window[curr_idx] = vec![0usize; d];
        }
        
        let mut i = 0; 
        while i < d && i < decoded_list.len() {
            self.window[curr_idx][i] = decoded_list[i];
            i += 1;
        }

        Some(self.curr as usize)
    }
} 

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
    BV: AsRef<BVGraph<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
>>> BVGraphNodeIterator<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
        BV
> {
    #[inline(always)]
    pub fn has_next(&self) -> bool {
        self.curr < self.n as i64 - 1
    }

    #[inline(always)]
    pub fn outdegree(&self) -> usize {
        assert!(self.curr != self.from as i64 - 1);
        self.outd[self.curr as usize % self.cyclic_buffer_size]
    }

    #[inline(always)]
    pub fn successor_array(&self) -> &[usize] {
        assert!(self.curr != self.from as i64 - 1);
        self.window[self.curr as usize % self.cyclic_buffer_size].as_slice()
    }
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> AsMut<BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
>> for BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
> {

    fn as_mut(&mut self) -> &mut BVGraph<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
    > {
        self
    }
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> AsRef<BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
>> for BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
>
{
    fn as_ref(&self) -> &BVGraph<
        InBlockCoding, 
        InBlockCountCoding, 
        InOutdegreeCoding, 
        InOffsetCoding, 
        InReferenceCoding,
        InIntervalCoding, 
        InResidualCoding, 
        OutBlockCoding, 
        OutBlockCountCoding, 
        OutOutdegreeCoding, 
        OutOffsetCoding, 
        OutReferenceCoding,
        OutIntervalCoding, 
        OutResidualCoding
    > {
        self
    }
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> IntoIterator for BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
> 
{
    type Item = usize;

    type IntoIter = BVGraphNodeIterator<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
        BVGraph<
            InBlockCoding, 
            InBlockCountCoding, 
            InOutdegreeCoding, 
            InOffsetCoding, 
            InReferenceCoding,
            InIntervalCoding, 
            InResidualCoding, 
            OutBlockCoding, 
            OutBlockCountCoding, 
            OutOutdegreeCoding, 
            OutOffsetCoding, 
            OutReferenceCoding,
            OutIntervalCoding, 
            OutResidualCoding
        >
    >;

    fn into_iter(self) -> Self::IntoIter {
        BVGraphNodeIterator {
            n: self.n,
            ibs: BinaryReader::new(self.graph_memory.clone()),
            cyclic_buffer_size: self.in_window_size + 1,
            window: vec![vec![0usize; self.in_window_size + 1]; 1024],
            outd: vec![0usize; self.in_window_size + 1],
            graph: self,
            from: 0,
            curr: -1,
            _phantom_in_block_coding: PhantomData,
            _phantom_in_block_count_coding: PhantomData,
            _phantom_in_outdegree_coding: PhantomData,
            _phantom_in_offset_coding: PhantomData,
            _phantom_in_reference_coding: PhantomData,
            _phantom_in_interval_coding: PhantomData,
            _phantom_in_residual_coding: PhantomData,
            _phantom_out_block_coding: PhantomData,
            _phantom_out_block_count_coding: PhantomData,
            _phantom_out_outdegree_coding: PhantomData,
            _phantom_out_offset_coding: PhantomData,
            _phantom_out_reference_coding: PhantomData,
            _phantom_out_interval_coding: PhantomData,
            _phantom_out_residual_coding: PhantomData,
        }
    }
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
> {
    pub fn iter(&self) -> BVGraphNodeIterator<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
        &Self
    > {
        BVGraphNodeIterator {
            n: self.n,
            graph: self,
            ibs: BinaryReader::new(self.graph_memory.clone()),
            cyclic_buffer_size: self.in_window_size + 1,
            window: vec![vec![0usize; self.in_window_size + 1]; 1024],
            outd: vec![0usize; self.in_window_size + 1],
            from: 0,
            curr: -1,
            _phantom_in_block_coding: PhantomData,
            _phantom_in_block_count_coding: PhantomData,
            _phantom_in_outdegree_coding: PhantomData,
            _phantom_in_offset_coding: PhantomData,
            _phantom_in_reference_coding: PhantomData,
            _phantom_in_interval_coding: PhantomData,
            _phantom_in_residual_coding: PhantomData,
            _phantom_out_block_coding: PhantomData,
            _phantom_out_block_count_coding: PhantomData,
            _phantom_out_outdegree_coding: PhantomData,
            _phantom_out_offset_coding: PhantomData,
            _phantom_out_reference_coding: PhantomData,
            _phantom_out_interval_coding: PhantomData,
            _phantom_out_residual_coding: PhantomData,
        }
    }

    #[inline(always)]
    fn outdegree_internal(&self, x: usize) -> usize {
        if self.cached_node.get().is_some() && x == self.cached_node.get().unwrap() {
            return self.cached_outdegree.get().unwrap();
        }
        
        self.outdegrees_binary_wrapper.borrow_mut().position(self.offsets[x] as u64); // TODO: offsets are encoded
        let d = InOutdegreeCoding::read_next(&mut self.outdegrees_binary_wrapper.borrow_mut(), self.in_zeta_k) as usize;

        self.cached_node.set(Some(x));
        self.cached_outdegree.set(Some(d));
        self.cached_ptr.set(Some(self.outdegrees_binary_wrapper.borrow().get_position()));

        d
    }
    
    #[inline(always)]
    fn decode_list(&self, x: usize, decoder: &mut BinaryReader, window: Option<&mut Vec<Vec<usize>>>, outd: &mut [usize]) -> Vec<usize> {
        let cyclic_buffer_size = self.in_window_size + 1;
        let degree;
        if window.is_none() {
            degree = self.outdegree_internal(x);
            decoder.position(self.cached_ptr.get().unwrap() as u64);
        } else {
            degree = InOutdegreeCoding::read_next(decoder, self.in_zeta_k) as usize;
            outd[x % cyclic_buffer_size] = degree; 
        }

        if degree == 0 {
            return Vec::new();
        }

        let mut reference = -1;
        if self.in_window_size > 0 {
            reference = InReferenceCoding::read_next(decoder, self.in_zeta_k) as i64;
        }

        // Position in the circular buffer of the reference of the current node
        let reference_index = ((x as i64 - reference + cyclic_buffer_size as i64) as usize) % cyclic_buffer_size;

        let mut block = Vec::default();

        let mut extra_count;

        if reference > 0 {
            let block_count = InBlockCountCoding::read_next(decoder, self.in_zeta_k) as usize;
            if block_count != 0 {
                block = Vec::with_capacity(block_count);
            }

            let mut copied = 0; // # of copied successors
            let mut total = 0; // total # of successors specified in some copy block

            let mut i = 0;
            while i < block_count {
                block.push(InBlockCoding::read_next(decoder, self.in_zeta_k) as usize + if i == 0 {0} else {1});
                total += block[i];
                if (i & 1) == 0 { // Alternate, count only even blocks
                    copied += block[i];
                }

                i += 1;
            }

            // If the block count is even, we must compute the number of successors copied implicitly
            if (block_count & 1) == 0 {
                copied += (
                    if window.is_some() {outd[reference_index]} 
                    else {self.outdegree_internal((x as i64 - reference) as usize)}
                ) - total;
            }
            
            extra_count = degree - copied;
        } else {
            extra_count = degree;
        }

        let mut interval_count = 0; // Number of intervals

        let mut left = Vec::default();
        let mut len = Vec::default();

        if extra_count > 0 && self.in_min_interval_len != 0 {
            interval_count = GammaCode::read_next(decoder, self.in_zeta_k) as usize;
            
            if interval_count != 0 {
                left = Vec::with_capacity(interval_count);
                len = Vec::with_capacity(interval_count);
                
                left.push(nat2int(InIntervalCoding::read_next(decoder, self.in_zeta_k)) + x as i64);
                len.push(InIntervalCoding::read_next(decoder, self.in_zeta_k) as usize + self.in_min_interval_len);
                let mut prev = left[0] + len[0] as i64;  // Holds the last integer in the last interval
                extra_count -= len[0];

                let mut i = 1;
                while i < interval_count {
                    prev += InIntervalCoding::read_next(decoder, self.in_zeta_k) as i64 + 1;
                    
                    left.push(prev);
                    len.push(InIntervalCoding::read_next(decoder, self.in_zeta_k) as usize + self.in_min_interval_len);

                    prev += len[i] as i64;
                    extra_count -= len[i];

                    i += 1;
                }
            }
        }

        let mut residual_list = Vec::with_capacity(extra_count);
        if extra_count > 0 {
            residual_list.push(x as i64 + nat2int(InResidualCoding::read_next(decoder, self.in_zeta_k)));
            let mut remaining = extra_count - 1;
            let mut curr_len = 1;

            while remaining > 0 {
                residual_list.push(residual_list[curr_len - 1] + InResidualCoding::read_next(decoder, self.in_zeta_k) as i64 + 1);
                curr_len += 1;

                remaining -= 1;
            }
        }

        // The extra part is made by the contribution of intervals, if any, and by the residuals list.
        let mut extra_list;
        if interval_count > 0 {
            let total_lenght = len.iter().sum();

            extra_list = Vec::with_capacity(total_lenght);
            let mut curr_left = if !left.is_empty() {left[0]} else {0};
            let mut curr_index = 0;
            let mut curr_interval = 0;
            let mut remaining = left.len();

            while remaining > 0 {
                extra_list.push(curr_left + curr_index as i64);
                curr_index += 1;

                if curr_index == len[curr_interval] {
                    remaining -= 1;
                    if remaining != 0 {
                        curr_interval += 1;
                        curr_left = left[curr_interval];
                    }
                    curr_index = 0;
                }
            }           
        
            if extra_count > 0 {
                let len_residual = residual_list.len();
                let len_extra = extra_list.len();

                let mut temp_list = Vec::with_capacity(len_residual + len_extra);
                let mut idx0 = 0;
                let mut idx1 = 0;
                while idx0 < len_residual && idx1 < len_extra {
                    if residual_list[idx0] <= extra_list[idx1] {
                        temp_list.push(residual_list[idx0]);
                        idx0 += 1;
                    } else {
                        temp_list.push(extra_list[idx1]);
                        idx1 += 1;
                        
                    }
                }

                while idx0 < len_residual {
                    temp_list.push(residual_list[idx0]);
                    idx0 += 1;
                }

                while idx1 < len_extra {
                    temp_list.push(extra_list[idx1]);
                    idx1 += 1;
                }

                extra_list = temp_list;
            }
        } else {
            extra_list = residual_list;
        }

        let mut block_list = Vec::default();
        if reference > 0 {
            let decoded_reference;

            let mut reference_it = 
                if let Some(window) = window {
                    window[reference_index][0..outd[reference_index]].iter()
                } else {
                    decoded_reference = self.decode_list(
                        (x as i64 - reference) as usize, 
                        decoder,
                        None, 
                        &mut []
                    );
                    decoded_reference.iter()
                };
            
            let mask_len = block.len();
            let mut curr_mask = 0;
            let mut left;

            if mask_len != 0 {
                left = block[curr_mask] as i64;
                curr_mask += 1;
                if left == 0 && curr_mask < mask_len {
                    reference_it.nth(block[curr_mask] - 1);
                    curr_mask += 1;

                    left = if curr_mask < mask_len {curr_mask += 1; block[curr_mask - 1] as i64} else {-1};
                }
            } else {
                left = -1;
            }

            block_list = Vec::with_capacity(reference_it.len());

            while left != 0 {
                let next = reference_it.next();

                if next.is_none() {
                    break;
                }

                if left == -1 {
                    block_list.push(*next.unwrap());
                }
                
                if left > 0 {
                    left -= 1;
                    if left == 0 && curr_mask < mask_len {
                        reference_it.nth(block[curr_mask] - 1);
                        curr_mask += 1;

                        left = if curr_mask < mask_len {curr_mask += 1; block[curr_mask - 1] as i64} else {-1};
                    }
                    block_list.push(*next.unwrap());
                }                
            }
        }

        if reference <= 0 {
            let extra_list: Vec<usize> = extra_list.into_iter().map(|x| x as usize).collect();
            return extra_list;
        } else if extra_list.is_empty() {
            return block_list;
        };

        let len_block = block_list.len();
        let len_extra = extra_list.len();

        let mut temp_list = Vec::with_capacity(len_block + len_extra);
        let mut idx0 = 0;
        let mut idx1 = 0;
        while idx0 < len_block && idx1 < len_extra {
            if block_list[idx0] < extra_list[idx1] as usize {
                temp_list.push(block_list[idx0]);
                idx0 += 1;
            } else {
                temp_list.push(extra_list[idx1] as usize);
                idx1 += 1;
            }
        }

        while idx0 < len_block {
            temp_list.push(block_list[idx0]);
            idx0 += 1;
        }

        while idx1 < len_extra {
            temp_list.push(extra_list[idx1] as usize);
            idx1 += 1;
        }

        temp_list
    }

    #[inline(always)]
    fn add_cost(&self, ctx: usize, val: usize, cost: &mut f64, sym_cost: &[f64]) {
        let (token, nbits, _) = zuck_encode(val, K_ZUCK, I_ZUCK, J_ZUCK);
        debug_assert!(token < K_NUM_SYMBOLS);
        *cost += sym_cost[ctx * K_NUM_SYMBOLS + token] + nbits as f64;
    }

    #[inline(always)]
    fn remove_cost(&self, cost: &mut f64, sym_cost: &[f64]) {
        *cost -= sym_cost[RESIDUALS_BASE_CTX * K_NUM_SYMBOLS];
    }

    #[inline(always)]
    fn process_residuals(&self, residuals: &[usize], i: usize, adj_block: &[usize], cost: &mut f64, sym_cost: &mut [f64]) {
        let mut r = i;
        let mut last_delta = 0;
        let mut adj_pos = 0;
        let adj_lim = adj_block.len();
        let mut zero_run = 0;

        for (j, res) in residuals.iter().enumerate() {
            let mut ctx = 0;
            if j == 0 {
                ctx = FIRST_RESIDUAL_BASE_CTX + 
                    zuck_encode(residuals.len(), K_ZUCK, I_ZUCK, J_ZUCK)
                    .0
                    .min(NUM_FIRST_RESIDUAL_CTX - 1);
                last_delta = int2nat(*res as i64 - r as i64) as usize;
            } else {
                ctx = RESIDUALS_BASE_CTX +
                    zuck_encode(last_delta, K_ZUCK, I_ZUCK, J_ZUCK)
                    .0
                    .min(NUM_RESIDUAL_CTX - 1);
                last_delta = res - r;
                while adj_pos < adj_lim && adj_block[adj_pos] < r {
                    adj_pos += 1;
                }
                while adj_pos < adj_lim && adj_block[adj_pos] < *res {
                    debug_assert!(last_delta > 0);
                    last_delta -= 1;
                    adj_pos += 1;
                }
            }
            if last_delta != 0 {
                if zero_run >= self.out_min_interval_len {
                    for _ in self.out_min_interval_len..zero_run {
                        self.remove_cost(cost, sym_cost);
                    }
                    self.add_cost(RLE_CTX, zero_run - self.out_min_interval_len, cost, sym_cost);
                }
                zero_run = 0;
            }
            if last_delta == 0 {
                zero_run += 1;
            }
            self.add_cost(ctx, last_delta, cost, sym_cost);
            r = res + 1;
        }
        if zero_run >= self.out_min_interval_len {
            for _ in self.out_min_interval_len..zero_run {
                self.remove_cost(cost, sym_cost);
            }
            self.add_cost(RLE_CTX, zero_run - self.out_min_interval_len, cost, sym_cost);
        }
    }

    #[inline(always)]
    fn compute_blocks_and_residuals(&self, curr_list: &[usize], ref_list: &[usize], blocks: &mut Vec<usize>, residuals: &mut Vec<usize>) {
        blocks.clear();
        residuals.clear();
        
        let mut ipos = 0;
        let mut rpos = 0;
        let mut is_same = true;

        blocks.push(0);

        while ipos < curr_list.len() && rpos < ref_list.len() {
            let a = curr_list[ipos];
            let b = ref_list[rpos];

            match a.cmp(&b) {
                Ordering::Equal => {
                    ipos += 1;
                    rpos += 1;

                    if !is_same {
                        blocks.push(0);
                    }

                    *blocks.last_mut().unwrap() += 1;
                    is_same = true;
                }
                Ordering::Less => {
                    ipos += 1;
                    residuals.push(a);
                }
                Ordering::Greater => {
                    if is_same {
                        blocks.push(0);
                    }

                    *blocks.last_mut().unwrap() += 1;
                    is_same = false;
                    rpos += 1;
                }
            }
        }

        if ipos != curr_list.len() {
            for j in ipos..curr_list.len() {
                residuals.push(curr_list[j]);
            }
        }

        residuals.sort();

        if rpos == ref_list.len() || !is_same {
            blocks.pop();
        }
    }

    #[inline(always)]
    fn process_blocks(&self, blocks: &[usize], ref_list: &[usize], adj_block: &mut Vec<usize>, cost: &mut f64, sym_cost: &[f64]) {
        self.add_cost(BLOCK_COUNT_CTX, blocks.len(), cost, sym_cost);
        let mut copy = true;
        let mut pos = 0;

        for (j, blk) in blocks.iter().enumerate() {
            let mut b = *blk;
            if j > 0 {
                b -= 1;
            }

            let ctx = if j == 0 {BLOCK_CTX} else if j % 2 == 0 {BLOCK_CTX_EVEN} else {BLOCK_CTX_ODD};
            self.add_cost(ctx, b, cost, sym_cost);

            if copy {
                for _ in 0..*blk {
                    adj_block.push(ref_list[pos]);
                    pos += 1;
                }
            } else {
                pos += blk;
            }

            copy = !copy;
        }

        if copy {
            for _ in pos..ref_list.len() {
                adj_block.push(ref_list[pos]);
                pos += 1;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn compress(&mut self, graph_obs: &mut BinaryWriter, offsets_obs: &mut BinaryWriter) {
        let mut bit_offset: usize = 0;
        
        let cyclic_buffer_size = self.out_window_size + 1;
        // Cyclic array of previous lists
        let mut list = vec![vec![0; 1024]; cyclic_buffer_size];
        // The length of each list
        let mut list_len = vec![0; cyclic_buffer_size];
        // The depth of the references of each list
        let mut ref_count: Vec<i32> = vec![0; cyclic_buffer_size];

        // List of (best_candidate, best_reference) tuples which prevents recomputing the best candidate for each node
        let mut best_candidates = vec![(0, 0); self.n];
        let mut saved_costs = vec![0.0; self.n];

        let mut sym_cost = vec![1_f64; NUM_CONTEXTS * K_NUM_SYMBOLS];

        let mut residuals = Vec::new();
        let mut blocks = Vec::new();
        let mut adj_block = Vec::new();

        let mut node_iter = self.iter();

        const V: Vec<usize> = Vec::new();
        let mut values = [V; NUM_CONTEXTS];

        // Populate the above vectors with their respective values
        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.outdegree();
            let curr_idx = curr_node % cyclic_buffer_size;

            let ctx =  
                if curr_node == 0 || curr_node % 32 == 0 {
                    FIRST_DEGREE_CTX
                } else { 
                    DEGREE_BASE_CTX + zuck_encode(curr_node % 32, K_ZUCK, I_ZUCK, J_ZUCK).0.min(NUM_DEGREE_CTX - 1) 
                };
            values[ctx].push(outd);

            if outd > list[curr_idx].len() {
                list[curr_idx].resize(outd, 0);
            }
            
            list[curr_idx] = Vec::from(&node_iter.successor_array()[..outd]);
            list_len[curr_idx] = outd;
            
            if outd > 0 {
                let mut cand;
                
                ref_count[curr_idx] = -1;

                let mut c = 0_f64;
                self.process_residuals(list[curr_idx].as_slice(), curr_node, &adj_block, &mut c, sym_cost.as_mut_slice());
                let mut cost = c;
                let base_cost = c;
                saved_costs[curr_node] = 0.0;

                for r in 1..cyclic_buffer_size {
                    cand = ((curr_node + cyclic_buffer_size - r) % cyclic_buffer_size) as i32;
                    if ref_count[cand as usize] < (self.out_max_ref_count as i32) && list_len[cand as usize] != 0 {
                        adj_block.clear();
                        c = 0.0;
                        self.compute_blocks_and_residuals(&list[curr_idx], &list[cand as usize], &mut blocks, &mut residuals);
                        self.process_blocks(&blocks, &list[cand as usize], &mut adj_block, &mut c, sym_cost.as_mut_slice());
                        self.process_residuals(&residuals, curr_node, &adj_block, &mut c, sym_cost.as_mut_slice());

                        if c + 1e-6 < cost {
                            best_candidates[curr_node] = (cand as usize, r);
                            saved_costs[curr_node] = base_cost - c;
                            cost = c;
                        }
                    }
                }
                
                ref_count[curr_idx] = ref_count[best_candidates[curr_node].0] + 1;

                self.add_vals( 
                    curr_node, 
                    best_candidates[curr_node].1 as usize, 
                    list[best_candidates[curr_node].0].as_slice(), 
                    list[curr_idx].as_slice(),
                    &mut values,
                );
            }
        }

        debug_assert_eq!(graph_obs.written_bits, 0);

        for (i, ctx) in values.iter().enumerate() {
            if ctx.is_empty() {
                println!("context {i} is empty");
            }
        }

        // Create Huffman encoder
        let mut huff = HuffmanEncoder::new();

        // Write Huffman headers
        huff.init(&values, graph_obs);

        println!("Headers took {} bits", graph_obs.written_bits);
        
        // Now, compress each node
        node_iter = self.iter();        
        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.outdegree();
            let curr_idx = curr_node % cyclic_buffer_size;
            
            // println!("Curr node: {}, outdegree: {}", curr_node, outd);
            
            // We write the final offset to the offsets stream
            self.write_offset(offsets_obs, graph_obs.written_bits - bit_offset).unwrap();
            
            bit_offset = graph_obs.written_bits;
            
            let ctx = 
                if curr_node == 0 || curr_node % 32 == 0 {
                    FIRST_DEGREE_CTX
                } else { 
                    DEGREE_BASE_CTX + zuck_encode(curr_node % 32, K_ZUCK, I_ZUCK, J_ZUCK).0.min(NUM_DEGREE_CTX - 1) 
                };
            // Encode through Huffman
            huff.write_next(outd, graph_obs, ctx);

            if outd > list[curr_idx].len() {
                list[curr_idx].resize(outd, 0);
            }
            
            list[curr_idx] = Vec::from(&node_iter.successor_array()[..outd]);
            list_len[curr_idx] = outd;
            
            if outd > 0 {
                let (best_cand, best_ref) = best_candidates[curr_node];
                
                ref_count[curr_idx] = ref_count[best_cand] + 1;
                self.diff_comp(
                    graph_obs, 
                    curr_node, 
                    best_ref,
                    list[best_cand].as_slice(), 
                    list[curr_idx].as_slice(),
                    &huff,
                );
            }
        }

        self.write_offset(offsets_obs, graph_obs.written_bits - bit_offset).unwrap();
    }
    
    #[inline(always)]
    fn diff_comp(
        &self,
        graph_obs: &mut BinaryWriter,
        curr_node: usize,  
        reference: usize,
        ref_list: &[usize],
        curr_list: &[usize],
        huff: &HuffmanEncoder,
    ) {
        let mut blocks = Vec::new();
        let mut residuals = Vec::new();
        let mut adj_block = Vec::new();
        
        self.write_reference(graph_obs, reference);

        if reference != 0 {
            self.compute_blocks_and_residuals(curr_list, ref_list, &mut blocks, &mut residuals);

            // Process blocks
            huff.write_next(blocks.len(), graph_obs, BLOCK_COUNT_CTX);

            let mut copy = true;
            let mut pos = 0;

            for (j, blk) in blocks.iter().enumerate() {
                let mut b = *blk;
                if j > 0 {
                    b -= 1;
                }

                let ctx = if j == 0 {BLOCK_CTX} else if j % 2 == 0 {BLOCK_CTX_EVEN} else {BLOCK_CTX_ODD};
                huff.write_next(b, graph_obs, ctx);

                if copy {
                    for _ in 0..*blk {
                        adj_block.push(ref_list[pos]);
                        pos += 1;
                    }
                } else {
                    pos += blk;
                }

                copy = !copy;
            }

            if copy {
                for _ in pos..ref_list.len() {
                    adj_block.push(ref_list[pos]);
                    pos += 1;
                }
            }            
        } else {
            residuals = Vec::from(curr_list);
        }

        let mut res_ctxs = Vec::new();
        let mut res_vals = Vec::new();

        // Process residuals
        let mut r = curr_node;
        let mut last_delta = 0;
        let mut adj_pos = 0;
        let adj_lim = adj_block.len();
        let mut zero_run = 0;

        for (j, res) in residuals.iter().enumerate() {
            let ctx;
            if j == 0 {
                ctx = FIRST_RESIDUAL_BASE_CTX + 
                    zuck_encode(residuals.len(), K_ZUCK, I_ZUCK, J_ZUCK)
                    .0
                    .min(NUM_FIRST_RESIDUAL_CTX - 1);
                last_delta = int2nat(*res as i64 - r as i64) as usize;
            } else {
                ctx = RESIDUALS_BASE_CTX +
                    zuck_encode(last_delta, K_ZUCK, I_ZUCK, J_ZUCK)
                    .0
                    .min(NUM_RESIDUAL_CTX - 1);
                last_delta = *res - r;
                while adj_pos < adj_lim && adj_block[adj_pos] < r {
                    adj_pos += 1;
                }
                while adj_pos < adj_lim && adj_block[adj_pos] < *res {
                    debug_assert!(last_delta > 0);
                    last_delta -= 1;
                    adj_pos += 1;
                }
            }
            if last_delta != 0 {
                if zero_run >= self.out_min_interval_len {
                    for _ in self.out_min_interval_len..zero_run {
                        res_ctxs.pop();
                        res_vals.pop();
                    }
                    res_ctxs.push(RLE_CTX);
                    res_vals.push(zero_run - self.out_min_interval_len);
                }
                zero_run = 0;
            }
            if last_delta == 0 {
                zero_run += 1;
            }
            res_ctxs.push(ctx);
            res_vals.push(last_delta);
            r = res + 1;
        }
        if zero_run >= self.out_min_interval_len {
            for _ in self.out_min_interval_len..zero_run {
                res_ctxs.pop();
                res_vals.pop();
            }
            res_ctxs.push(RLE_CTX);
            res_vals.push(zero_run - self.out_min_interval_len);
        }

        debug_assert_eq!(res_ctxs.len(), res_vals.len());
        for i in 0..res_ctxs.len() {
            huff.write_next(res_vals[i], graph_obs, res_ctxs[i]);
        }
    }


    #[inline(always)]
    fn add_vals(
        &self,
        curr_node: usize,  
        reference: usize,
        ref_list: &[usize],
        curr_list: &[usize],
        vals: &mut [Vec<usize>],
    ) {
        let mut blocks = Vec::new();
        let mut residuals = Vec::new();
        let mut adj_block = Vec::new();

        if reference != 0 {
            self.compute_blocks_and_residuals(curr_list, ref_list, &mut blocks, &mut residuals);

            // Process blocks
            vals[BLOCK_COUNT_CTX].push(blocks.len());

            let mut copy = true;
            let mut pos = 0;

            for (j, blk) in blocks.iter().enumerate() {
                let mut b = *blk;
                if j > 0 {
                    b -= 1;
                }

                let ctx = if j == 0 {BLOCK_CTX} else if j % 2 == 0 {BLOCK_CTX_EVEN} else {BLOCK_CTX_ODD};
                vals[ctx].push(b);

                if copy {
                    for _ in 0..*blk {
                        adj_block.push(ref_list[pos]);
                        pos += 1;
                    }
                } else {
                    pos += blk;
                }

                copy = !copy;
            }

            if copy {
                for _ in pos..ref_list.len() {
                    adj_block.push(ref_list[pos]);
                    pos += 1;
                }
            }
        } else {
            residuals = Vec::from(curr_list);
        }

        let mut res_ctxs = Vec::new();
        let mut res_vals = Vec::new();

        // Process residuals
        let mut r = curr_node;
        let mut last_delta = 0;
        let mut adj_pos = 0;
        let adj_lim = adj_block.len();
        let mut zero_run = 0;

        for (j, res) in residuals.iter().enumerate() {
            let ctx;
            if j == 0 {
                ctx = FIRST_RESIDUAL_BASE_CTX + 
                    zuck_encode(residuals.len(), K_ZUCK, I_ZUCK, J_ZUCK)
                    .0
                    .min(NUM_FIRST_RESIDUAL_CTX - 1);
                last_delta = int2nat(*res as i64 - r as i64) as usize;
            } else {
                ctx = RESIDUALS_BASE_CTX +
                    zuck_encode(last_delta, K_ZUCK, I_ZUCK, J_ZUCK)
                    .0
                    .min(NUM_RESIDUAL_CTX - 1);
                last_delta = res - r;
                while adj_pos < adj_lim && adj_block[adj_pos] < r {
                    adj_pos += 1;
                }
                while adj_pos < adj_lim && adj_block[adj_pos] < *res {
                    debug_assert!(last_delta > 0);
                    last_delta -= 1;
                    adj_pos += 1;
                }
            }
            if last_delta != 0 {
                if zero_run >= self.out_min_interval_len {
                    for _ in self.out_min_interval_len..zero_run {
                        res_ctxs.pop();
                        res_vals.pop();
                    }
                    res_ctxs.push(RLE_CTX);
                    res_vals.push(zero_run - self.out_min_interval_len);
                }
                zero_run = 0;
            }
            if last_delta == 0 {
                zero_run += 1;
            }
            res_ctxs.push(ctx);
            res_vals.push(last_delta);
            r = res + 1;
        }
        if zero_run >= self.out_min_interval_len {
            for _ in self.out_min_interval_len..zero_run {
                res_ctxs.pop();
                res_vals.pop();
            }
            res_ctxs.push(RLE_CTX);
            res_vals.push(zero_run - self.out_min_interval_len);
        }

        debug_assert_eq!(res_ctxs.len(), res_vals.len());
        for i in 0..res_ctxs.len() {
            vals[res_ctxs[i]].push(res_vals[i]);
        }
    }

    #[inline(always)]
    fn write_reference(&self, graph_obs: &mut BinaryWriter, reference: usize) -> Result<usize, String> {
        if reference > self.out_window_size {
            return Err("The required reference is incompatible with the window size".to_string());
        }

        OutReferenceCoding::write_next(graph_obs, reference as u64, self.out_zeta_k);
        Ok(reference)
    }

    #[inline(always)]
    fn write_block_count(&self, graph_obs: &mut BinaryWriter, block_count: usize) -> Result<usize, String> {
        OutBlockCountCoding::write_next(graph_obs, block_count as u64, self.out_zeta_k);
        Ok(block_count)
    }

    #[inline(always)]
    fn write_offset(&self, offset_obs: &mut BinaryWriter, offset: usize) -> Result<usize, String> {
        OutOffsetCoding::write_next(offset_obs, offset as u64, self.out_zeta_k);
        Ok(offset)
    }
}

pub struct BVGraphBuilder<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Rc<[u8]>,
    loaded_offsets: Box<[usize]>,
    graph_binary_wrapper: BinaryReader,
    outdegrees_binary_wrapper: BinaryReader,
    cached_node: Option<usize>,
    cached_outdegree: Option<usize>,
    cached_ptr: Option<usize>,
    in_max_ref_count: usize,
    in_window_size: usize,
    in_min_interval_len: usize,
    out_max_ref_count: usize,
    out_window_size: usize,
    out_min_interval_len: usize,
    in_zeta_k: Option<u64>,
    out_zeta_k: Option<u64>,
    _phantom_in_block_coding: PhantomData<InBlockCoding>,
    _phantom_in_block_count_coding: PhantomData<InBlockCountCoding>,
    _phantom_in_outdegree_coding: PhantomData<InOutdegreeCoding>,
    _phantom_in_offset_coding: PhantomData<InOffsetCoding>,
    _phantom_in_reference_coding: PhantomData<InReferenceCoding>,
    _phantom_in_interval_coding: PhantomData<InIntervalCoding>,
    _phantom_in_residual_coding: PhantomData<InResidualCoding>,
    _phantom_out_block_coding: PhantomData<OutBlockCoding>,
    _phantom_out_block_count_coding: PhantomData<OutBlockCountCoding>,
    _phantom_out_outdegree_coding: PhantomData<OutOutdegreeCoding>,
    _phantom_out_offset_coding: PhantomData<OutOffsetCoding>,
    _phantom_out_reference_coding: PhantomData<OutReferenceCoding>,
    _phantom_out_interval_coding: PhantomData<OutIntervalCoding>,
    _phantom_out_residual_coding: PhantomData<OutResidualCoding>,
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> Default for BVGraphBuilder<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
> {
    fn default() -> Self {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Rc::new([]), 
            loaded_offsets: Box::default(), 
            graph_binary_wrapper: BinaryReader::default(),
            outdegrees_binary_wrapper: BinaryReader::default(),
            cached_node: None, 
            cached_outdegree: None, 
            cached_ptr: None, 
            in_max_ref_count: 0, 
            in_window_size: 0, 
            in_min_interval_len: 0,
            out_max_ref_count: 3,
            out_window_size: 7,
            out_min_interval_len: 4,
            in_zeta_k: None,
            out_zeta_k: Some(3),
            _phantom_in_block_coding: PhantomData,
            _phantom_in_block_count_coding: PhantomData,
            _phantom_in_outdegree_coding: PhantomData,
            _phantom_in_offset_coding: PhantomData,
            _phantom_in_reference_coding: PhantomData,
            _phantom_in_interval_coding: PhantomData,
            _phantom_in_residual_coding: PhantomData,
            _phantom_out_block_coding: PhantomData,
            _phantom_out_block_count_coding: PhantomData,
            _phantom_out_outdegree_coding: PhantomData,
            _phantom_out_offset_coding: PhantomData,
            _phantom_out_reference_coding: PhantomData,
            _phantom_out_interval_coding: PhantomData,
            _phantom_out_residual_coding: PhantomData,
        }
    }
}

impl<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: Huffman,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: Huffman,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: Huffman,
    OutResidualCoding: Huffman,
> BVGraphBuilder<
    InBlockCoding,
    InBlockCountCoding,
    InOutdegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutdegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
> {
    pub fn new() -> BVGraphBuilder<
        InBlockCoding, InBlockCountCoding, InOutdegreeCoding, InOffsetCoding, InReferenceCoding, InIntervalCoding, InResidualCoding, 
        OutBlockCoding, OutBlockCountCoding, OutOutdegreeCoding, OutOffsetCoding, OutReferenceCoding, OutIntervalCoding, OutResidualCoding
    > {
        Self::default()
    }

    /// Loads a previously-compressed BVGraph.
    /// 
    /// This method can be called either before or after [`Self::load_offsets()`].
    ///  
    /// # Arguments
    /// 
    /// * `basename` - The base name of the compressed graph file
    /// 
    /// # Examples
    /// ```
    /// let file_base_name = "graph1";
    /// let builder = BVGraphBuilder::<
    ///     GammaCode, GammaCode, UnaryCode, DeltaCode, GammaCode, GammaCode, ZetaCode
    ///     GammaCode, GammaCode, UnaryCode, DeltaCode, GammaCode, GammaCode, ZetaCode
    /// >::new()
    ///     .load_properties(file_base_name);
    ///     .load_graph(file_base_name);
    /// ```
    pub fn load_graph(mut self, basename: &str) -> Self {
        let graph = fs::read(format!("{}.graph", basename)).unwrap();

        let graph = graph.into_boxed_slice().into();

        self.loaded_graph = graph;
        self.graph_binary_wrapper = BinaryReader::new(self.loaded_graph.clone());

        self
    }

    /// Loads a previously-compressed BVGraph's offsets file.
    /// 
    /// This method can be called either before or after [`Self::load_graph()`], but <strong>always</strong> after loading the offsets.
    ///  
    /// # Arguments
    /// 
    /// * `basename` - The base name of the compressed graph file
    /// 
    /// # Examples
    /// ```
    /// let file_base_name = "graph1";
    /// let builder = BVGraphBuilder::<
    ///     GammaCode, GammaCode, UnaryCode, DeltaCode, GammaCode, GammaCode, ZetaCode
    ///     GammaCode, GammaCode, UnaryCode, DeltaCode, GammaCode, GammaCode, ZetaCode
    /// >::new()
    ///     .load_properties(file_base_name);
    ///     .load_graph(file_base_name);
    ///     .load_offsets(file_base_name);
    /// let graph = builder.build();
    /// ```
    pub fn load_offsets(mut self, basename: &str) -> Self {
        assert!(self.num_nodes > 0, "The number of nodes has to be >0.");
        let offsets = fs::read(format!("{}.offsets", basename)).unwrap();

        let mut curr = 0;

        let mut offsets_ibs = BinaryReader::new(offsets.into());
        
        let mut n = self.num_nodes;

        let mut increasing_offsets = Vec::with_capacity(n);

        while n > 0 {
            curr += InOffsetCoding::read_next(&mut offsets_ibs, self.in_zeta_k);
            increasing_offsets.push(curr as usize);

            n -= 1;
        }

        self.loaded_offsets = increasing_offsets.into_boxed_slice();

        self
    }

    /// Creates a new binary wrapper around the previously-loaded graph.
    /// 
    /// This wrapper will be used only for operations concerning the outdegree of nodes.
    /// 
    /// This has to be called only after [`Self::load_graph()`].
    pub fn load_outdegrees(mut self) -> Self {
        self.outdegrees_binary_wrapper = BinaryReader::new(self.loaded_graph.clone());

        self 
    }

    /// Sets the maximum reference chain length for reading.
    /// 
    /// # Arguments
    /// 
    /// * `ref_count` - The maximum length of the chain.
    pub fn set_in_max_ref_count(mut self, ref_count: usize) -> Self {
        self.in_max_ref_count = ref_count;

        self
    }

    /// Sets the maximum reference window size for reading.
    /// 
    /// # Arguments
    /// 
    /// * `window_size` - The maximum length of the window.
    pub fn set_in_window_size(mut self, window_size: usize) -> Self {
        self.in_window_size = window_size;

        self
    }

    /// Sets the minimum length of the intervals for reading.
    /// 
    /// # Arguments
    /// 
    /// * `min_interval_length` - The minimum length of the intervals.
    pub fn set_in_min_interval_len(mut self, min_interval_len: usize) -> Self {
        self.in_min_interval_len = min_interval_len;

        self
    }

    // Sets the `k` parameter for reading *zeta*-codes, if present.
    /// 
    /// # Arguments
    /// 
    /// * `zk` - An option containing the value of *k*. If it is not `None` its value has to be >= 1.
    pub fn set_in_zeta(mut self, zk: Option<u64>) -> Self {
        self.in_zeta_k = zk;

        self
    }

    /// Sets the maximum reference chain length for writing.
    /// 
    /// # Arguments
    /// 
    /// * `ref_count` - The maximum length of the chain.
    pub fn set_out_max_ref_count(mut self, ref_count: usize) -> Self {
        self.out_max_ref_count = ref_count;

        self
    }

    /// Sets the maximum reference window size for writing.
    /// 
    /// # Arguments
    /// 
    /// * `window_size` - The maximum length of the window.
    pub fn set_out_window_size(mut self, window_size: usize) -> Self {
        self.out_window_size = window_size;

        self
    }

    /// Sets the minimum length of the intervals for writing.
    /// 
    /// # Arguments
    /// 
    /// * `min_interval_length` - The minimum length of the intervals.
    pub fn set_out_min_interval_len(mut self, min_interval_len: usize) -> Self {
        self.out_min_interval_len = min_interval_len;

        self
    }

    // Sets the `k` parameter for writing *zeta*-codes, if present.
    /// 
    /// # Arguments
    /// 
    /// * `zk` - An option containing the value of *k*. If it is not `None` its value has to be >= 1.
    pub fn set_out_zeta(mut self, zk: Option<u64>) -> Self {
        self.out_zeta_k = zk;

        self
    }

    /// Sets the number of nodes of the graph.
    /// 
    /// # Arguments
    /// 
    /// * `nodes` - The amount of nodes to set.
    pub fn set_num_nodes(mut self, nodes: usize) -> Self {
        self.num_nodes = nodes;

        self
    }

    /// Sets the number of edges of the graph.
    /// 
    /// # Arguments
    /// 
    /// * `edges` - The amount of edges to set.
    pub fn set_num_edges(mut self, edges: usize) -> Self {
        self.num_edges = edges;

        self
    }

    /// Constructs the BVGraph object.
    pub fn build(self) -> BVGraph<
        InBlockCoding,
        InBlockCountCoding,
        InOutdegreeCoding,
        InOffsetCoding,
        InReferenceCoding,
        InIntervalCoding,
        InResidualCoding,
        OutBlockCoding,
        OutBlockCountCoding,
        OutOutdegreeCoding,
        OutOffsetCoding,
        OutReferenceCoding,
        OutIntervalCoding,
        OutResidualCoding,
    > {
        BVGraph::<
        InBlockCoding, InBlockCountCoding, InOutdegreeCoding, InOffsetCoding, InReferenceCoding, InIntervalCoding, InResidualCoding, 
        OutBlockCoding, OutBlockCountCoding, OutOutdegreeCoding, OutOffsetCoding, OutReferenceCoding, OutIntervalCoding, OutResidualCoding
        > { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph, 
            offsets: self.loaded_offsets,
            graph_binary_wrapper: RefCell::new(self.graph_binary_wrapper),
            outdegrees_binary_wrapper: RefCell::new(self.outdegrees_binary_wrapper),
            cached_node: Cell::new(self.cached_node), 
            cached_outdegree: Cell::new(self.cached_outdegree), 
            cached_ptr: Cell::new(self.cached_ptr), 
            in_max_ref_count: self.in_max_ref_count, 
            in_window_size: self.in_window_size,
            in_min_interval_len: self.in_min_interval_len,
            out_max_ref_count: self.out_max_ref_count, 
            out_window_size: self.out_window_size,
            out_min_interval_len: self.out_min_interval_len,
            in_zeta_k: self.in_zeta_k,
            out_zeta_k: self.out_zeta_k,
            compression_vectors: CompressionVectors::default(),
            _phantom_in_block_coding: PhantomData,
            _phantom_in_block_count_coding: PhantomData,
            _phantom_in_outdegree_coding: PhantomData,
            _phantom_in_offset_coding: PhantomData,
            _phantom_in_reference_coding: PhantomData,
            _phantom_in_interval_coding: PhantomData,
            _phantom_in_residual_coding: PhantomData,
            _phantom_out_block_coding: PhantomData,
            _phantom_out_block_count_coding: PhantomData,
            _phantom_out_outdegree_coding: PhantomData,
            _phantom_out_offset_coding: PhantomData,
            _phantom_out_reference_coding: PhantomData,
            _phantom_out_interval_coding: PhantomData,
            _phantom_out_residual_coding: PhantomData,
        }
    }
}