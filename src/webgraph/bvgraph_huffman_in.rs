use std::{fs, vec, cmp::Ordering, marker::PhantomData, cell::{RefCell, Cell}, rc::Rc};

use sucds::{mii_sequences::{EliasFanoBuilder, EliasFano}, Serializable};

use crate::{ImmutableGraph, int2nat, nat2int, properties::Properties, utils::{encodings::{UniversalCode, GammaCode, Huffman, zuck_encode, K_ZUCK, I_ZUCK, J_ZUCK}, timers::Timer}, BitsLen, huffman_zuckerli::huffman_decoder::HuffmanDecoder};
use crate::bitstreams::{BinaryReader, BinaryWriterBuilder};

use super::bvgraph_huffman_out::{INTERVALS_LEN_IDX_BEGIN, INTERVALS_LEN_IDX_LEN, OUTD_IDX_BEGIN, BLOCKS_IDX_BEGIN, INTERVALS_LEFT_IDX_BEGIN, RESIDUALS_IDX_BEGIN};

#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub struct DecompressionStats {
    pub block_time: Timer,
    pub block_count_time: Timer,
    pub outdegree_time: Timer,
    pub reference_time: Timer,
    pub interval_time: Timer,
    pub interval_count_time: Timer,
    pub residual_time: Timer,
}

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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
> {
    n: usize,
    m: usize,
    pub graph_memory: Rc<[u8]>,
    pub offsets: Box<[usize]>,
    pub graph_binary_wrapper: Rc<RefCell<BinaryReader>>,
    outdegrees_binary_wrapper: RefCell<BinaryReader>,
    cached_node: Cell<Option<usize>>,
    cached_outdegree: Cell<Option<usize>>,
    cached_ptr: Cell<Option<usize>>,
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: Option<u64>,
    compression_vectors: CompressionVectors,
    pub decompression_stats: RefCell<DecompressionStats>,
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize> {
        todo!("Impossible Huffman with this signature")
    }

    fn store(&mut self, basename: &str) -> std::io::Result<()> {      
        let mut graph_obs = BinaryWriterBuilder::new();
        let mut offsets_obs = BinaryWriterBuilder::new();

        self.compress(&mut graph_obs, &mut offsets_obs);
        
        let mut out_stats = String::new();
            
        out_stats.push_str("################### Sequential Huffman decompression stats ###################\n");
        out_stats.push_str(&format!("time outdegrees {} ns\n", self.decompression_stats.borrow_mut().outdegree_time.total_time));
        out_stats.push_str(&format!("time blocks {} ns\n", self.decompression_stats.borrow_mut().block_time.total_time));
        out_stats.push_str(&format!("time block count {} ns\n", self.decompression_stats.borrow_mut().block_count_time.total_time));
        out_stats.push_str(&format!("time references {} ns\n", self.decompression_stats.borrow_mut().reference_time.total_time));
        out_stats.push_str(&format!("time interval count {} ns\n", self.decompression_stats.borrow_mut().interval_count_time.total_time));
        out_stats.push_str(&format!("time intervals {} ns\n", self.decompression_stats.borrow_mut().interval_time.total_time));
        out_stats.push_str(&format!("time residuals {} ns\n", self.decompression_stats.borrow_mut().residual_time.total_time));
    
        fs::write(format!("{}.stats", basename), out_stats)?;

        Ok(())
    }
}

pub struct BVGraphNodeIterator<
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
    pub ibs: Rc<RefCell<BinaryReader>>,
    // An instance of `HuffmanDecoder` for decoding data
    pub huff_decoder: HuffmanDecoder,
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
        let decoded_list = 
            self.graph.as_ref().decode_list(
                self.curr as usize, 
                self.ibs.clone(), 
                Some(&mut self.window), 
                &mut self.outd,
                &mut self.huff_decoder,
                true
            );

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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
        let ibs = Rc::new(RefCell::new(BinaryReader::new(self.graph_memory.clone())));

        let mut huff_decoder = HuffmanDecoder::new();
        huff_decoder.decode_headers(&mut ibs.borrow_mut(), INTERVALS_LEN_IDX_BEGIN + INTERVALS_LEN_IDX_LEN);

        BVGraphNodeIterator {
            n: self.n,
            ibs,
            huff_decoder,
            cyclic_buffer_size: self.window_size + 1,
            window: vec![vec![0usize; self.window_size + 1]; 1024],
            outd: vec![0usize; self.window_size + 1],
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
        let ibs = Rc::new(RefCell::new(BinaryReader::new(self.graph_memory.clone())));

        let mut huff_decoder = HuffmanDecoder::new();
        huff_decoder.decode_headers(&mut ibs.borrow_mut(), INTERVALS_LEN_IDX_BEGIN + INTERVALS_LEN_IDX_LEN);

        BVGraphNodeIterator {
            n: self.n,
            graph: self,
            ibs,
            huff_decoder,
            cyclic_buffer_size: self.window_size + 1,
            window: vec![vec![0usize; self.window_size + 1]; 1024],
            outd: vec![0usize; self.window_size + 1],
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
    fn outdegree_internal(&self, x: usize, huff_outdegrees: &mut HuffmanDecoder, to_record: bool) -> usize {
        if self.cached_node.get().is_some() && x == self.cached_node.get().unwrap() {
            return self.cached_outdegree.get().unwrap();
        }
        
        self.outdegrees_binary_wrapper.borrow_mut().position(self.offsets[x] as u64);
        let d;
        if x == 0 || x % 32 == 0 {
            if to_record { self.decompression_stats.borrow_mut().outdegree_time.start(); }
            d = huff_outdegrees.read_next(&mut self.outdegrees_binary_wrapper.borrow_mut(), OUTD_IDX_BEGIN + 0);
            if to_record { self.decompression_stats.borrow_mut().outdegree_time.stop(); }
        } else {
            let ctx = 1 + zuck_encode((x % 32) + 1, K_ZUCK, I_ZUCK, J_ZUCK).0.min(30);
            if to_record { self.decompression_stats.borrow_mut().outdegree_time.start(); }
            d = huff_outdegrees.read_next(&mut self.outdegrees_binary_wrapper.borrow_mut(), OUTD_IDX_BEGIN + ctx);
            if to_record { self.decompression_stats.borrow_mut().outdegree_time.stop(); }
        }

        self.cached_node.set(Some(x));
        self.cached_outdegree.set(Some(d));
        self.cached_ptr.set(Some(self.outdegrees_binary_wrapper.borrow().get_position()));

        d
    }

    #[inline(always)]
    pub fn successors(
        &mut self, 
        x: usize,
        huff: &mut HuffmanDecoder,
    ) -> Box<[usize]> {
        assert!(x < self.n, "Node index out of range {}", x);
        self.decode_list(x, self.graph_binary_wrapper.clone(), None, &mut [], huff, true)
    }

    #[inline(always)]
    pub fn decode_list(
        &self, 
        x: usize, 
        decoder: Rc<RefCell<BinaryReader>>, 
        window: Option<&mut Vec<Vec<usize>>>, 
        outd: &mut [usize],
        huff: &mut HuffmanDecoder,
        to_record: bool
    ) -> Box<[usize]> {
        let cyclic_buffer_size = self.window_size + 1;
        let degree;
        if window.is_none() {
            degree = self.outdegree_internal(x, huff, to_record);
            decoder.borrow_mut().position(self.cached_ptr.get().unwrap() as u64);
        } else {
            let ctx = ////////////// Probably should use the context based on its predecessing node's outdegree. Is it stored in outd?
                if x == 0 || x % 32 == 0 {
                    0
                } else {
                    1 + zuck_encode((x % 32) + 1, K_ZUCK, I_ZUCK, J_ZUCK).0.min(30)
                };
            if to_record { self.decompression_stats.borrow_mut().outdegree_time.start(); }
            degree = huff.read_next(&mut decoder.borrow_mut(), OUTD_IDX_BEGIN + ctx);
            if to_record { self.decompression_stats.borrow_mut().outdegree_time.stop(); }
            outd[x % cyclic_buffer_size] = degree; 
        }

        if degree == 0 {
            return Box::new([]);
        }

        let mut reference = -1;
        if self.window_size > 0 {
            reference = InReferenceCoding::read_next(&mut decoder.borrow_mut(), self.zeta_k) as i64;
        }

        // Position in the circular buffer of the reference of the current node
        let reference_index = ((x as i64 - reference + cyclic_buffer_size as i64) as usize) % cyclic_buffer_size;

        let mut block = Vec::default();

        let mut extra_count;

        if reference > 0 {
            let block_count = InBlockCountCoding::read_next(&mut decoder.borrow_mut(), self.zeta_k) as usize;
            if block_count != 0 {
                block = Vec::with_capacity(block_count);
            }

            let mut copied = 0; // # of copied successors
            let mut total = 0; // total # of successors specified in some copy block

            let mut i = 0;
            while i < block_count {
                if to_record { self.decompression_stats.borrow_mut().block_time.start(); }
                let h = huff.read_next(&mut decoder.borrow_mut(), BLOCKS_IDX_BEGIN + if i == 0 {0} else {i % 2 + 1}); 
                if to_record { self.decompression_stats.borrow_mut().block_time.stop(); }
                block.push(h + if i == 0 {0} else {1});
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
                    else {self.outdegree_internal((x as i64 - reference) as usize, huff, to_record)} //////////// The same here. It should use its predecessor node's outd
                ) - total;
            }
            
            extra_count = degree - copied;
        } else {
            extra_count = degree;
        }

        let mut interval_count = 0; // Number of intervals

        let mut left = Vec::default();
        let mut len = Vec::default();

        if extra_count > 0 && self.min_interval_len != 0 {
            interval_count = GammaCode::read_next(&mut decoder.borrow_mut(), self.zeta_k) as usize;
            
            if interval_count != 0 {
                left = Vec::with_capacity(interval_count);
                len = Vec::with_capacity(interval_count);
                
                if to_record { self.decompression_stats.borrow_mut().interval_time.start(); }
                let mut prev_left = huff.read_next(&mut decoder.borrow_mut(), INTERVALS_LEFT_IDX_BEGIN);
                let mut prev_len = huff.read_next(&mut decoder.borrow_mut(), INTERVALS_LEN_IDX_BEGIN);
                if to_record { self.decompression_stats.borrow_mut().interval_time.stop(); }

                left.push(nat2int(prev_left as u64) + x as i64);
                len.push(prev_len + self.min_interval_len);
                let mut prev = left[0] + len[0] as i64;  // Holds the last integer in the last interval
                extra_count -= len[0];

                let mut i = 1;
                while i < interval_count {
                    let mut ctx = 1 + 
                        zuck_encode(prev_left, K_ZUCK, I_ZUCK, J_ZUCK)
                        .0
                        .min(30);
                    if to_record { self.decompression_stats.borrow_mut().interval_time.start(); }
                    prev_left = huff.read_next(&mut decoder.borrow_mut(), INTERVALS_LEFT_IDX_BEGIN + ctx);
                    if to_record { self.decompression_stats.borrow_mut().interval_time.stop(); }
                    prev += prev_left as i64 + 1;                    
                    left.push(prev);

                    ctx = 1 +
                        zuck_encode(prev_len, K_ZUCK, I_ZUCK, J_ZUCK)
                        .0
                        .min(30);
                    if to_record { self.decompression_stats.borrow_mut().interval_time.start(); }
                    prev_len = huff.read_next(&mut decoder.borrow_mut(), INTERVALS_LEN_IDX_BEGIN + ctx);
                    if to_record { self.decompression_stats.borrow_mut().interval_time.stop(); }
                    len.push(prev_len + self.min_interval_len);

                    prev += len[i] as i64;
                    extra_count -= len[i];

                    i += 1;
                }
            }
        }

        let mut residual_list = Vec::with_capacity(extra_count);
        if extra_count > 0 {
            let mut ctx = 
                zuck_encode(extra_count, K_ZUCK, I_ZUCK, J_ZUCK)
                .0
                .min(31);
            if to_record { self.decompression_stats.borrow_mut().residual_time.start(); }
            let mut prev_residual = huff.read_next(&mut decoder.borrow_mut(), RESIDUALS_IDX_BEGIN + ctx);
            if to_record { self.decompression_stats.borrow_mut().residual_time.stop(); }
            residual_list.push(x as i64 + nat2int(prev_residual as u64));
            let mut remaining = extra_count - 1;
            let mut curr_len = 1;

            while remaining > 0 {
                ctx = 32 +
                    zuck_encode(prev_residual, K_ZUCK, I_ZUCK, J_ZUCK)
                    .0
                    .min(79);
                if to_record { self.decompression_stats.borrow_mut().residual_time.start(); }
                prev_residual = huff.read_next(&mut decoder.borrow_mut(), RESIDUALS_IDX_BEGIN + ctx);
                if to_record { self.decompression_stats.borrow_mut().residual_time.stop(); }
                residual_list.push(residual_list[curr_len - 1] + prev_residual as i64 + 1);
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
                        &mut [],
                        huff,
                        false
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
            return extra_list.into_boxed_slice();
        } else if extra_list.is_empty() {
            return block_list.into_boxed_slice();
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

        temp_list.into_boxed_slice()
    }

    #[inline(always)]
    pub fn compress(&mut self, graph_obs: &mut BinaryWriterBuilder, offsets_obs: &mut BinaryWriterBuilder) {
        let mut bit_offset: usize = 0;
        
        let mut bit_count = BinaryWriterBuilder::new();
        
        let cyclic_buffer_size = self.window_size + 1;
        // Cyclic array of previous lists
        let mut list = vec![vec![0; 1024]; cyclic_buffer_size];
        // The length of each list
        let mut list_len = vec![0; cyclic_buffer_size];
        // The depth of the references of each list
        let mut ref_count: Vec<i32> = vec![0; cyclic_buffer_size];
        
        let mut node_iter = self.iter();
        
        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.outdegree();
            let curr_idx = curr_node % cyclic_buffer_size;
            
            // println!("Curr node: {}, outdegree: {}", curr_node, outd);
            
            // We write the final offset to the offsets stream
            self.write_offset(offsets_obs, graph_obs.written_bits - bit_offset).unwrap();
            
            bit_offset = graph_obs.written_bits;
            
            self.write_outdegree(graph_obs, outd).unwrap();
            
            if outd > list[curr_idx].len() {
                list[curr_idx].resize(outd, 0);
            }
            
            list[curr_idx] = Vec::from(&node_iter.successor_array()[..outd]);
            list_len[curr_idx] = outd;
            
            if outd > 0 {
                let mut best_comp = i64::MAX;
                let mut best_cand = -1;
                let mut best_ref: i32 = -1;
                let mut cand;
                
                ref_count[curr_idx] = -1;

                for r in 0..cyclic_buffer_size {
                    cand = ((curr_node + cyclic_buffer_size - r) % cyclic_buffer_size) as i32;
                    if ref_count[cand as usize] < (self.max_ref_count as i32) && list_len[cand as usize] != 0 {
                        let diff_comp = 
                            self.diff_comp(&mut bit_count, 
                                            curr_node, 
                                            r, 
                                            list[cand as usize].as_slice(), 
                                            list[curr_idx].as_slice()
                            ).unwrap();
                        if (diff_comp as i64) < best_comp {
                            best_comp = diff_comp as i64;
                            best_cand = cand;
                            best_ref = r as i32;
                        }
                    }
                }
                                    
                debug_assert!(best_cand >= 0);
                
                ref_count[curr_idx] = ref_count[best_cand as usize] + 1;
                self.diff_comp(
                    graph_obs, 
                    curr_node, 
                    best_ref as usize, 
                    list[best_cand as usize].as_slice(), 
                    list[curr_idx].as_slice(),
                ).unwrap();
            }
        }

        self.write_offset(offsets_obs, graph_obs.written_bits - bit_offset).unwrap();
    }

    #[inline(always)]
    fn intervalize(
        &self,
        extras: &Vec<usize>,
        left: &mut Vec<usize>,
        len: &mut Vec<usize>,
        residuals: &mut Vec<usize>
    ) -> usize {
        let mut n_interval = 0;
        let v_len = extras.len();

        let mut j;

        left.clear();
        len.clear();
        residuals.clear();

        let mut i = 0;

        while i < v_len {
            j = 0;
            if i < v_len - 1 && extras[i] + 1 == extras[i + 1] {
                j += 1;
                while i + j < v_len - 1 && extras[i + j] + 1 == extras[i + j + 1] {
                    j += 1;
                }
                j += 1;

                // Now j is the # of integers in the interval
                if j >= self.min_interval_len {
                    left.push(extras[i]);
                    len.push(j);
                    n_interval += 1;
                    i += j - 1;
                }
            }

            if j < self.min_interval_len {
                residuals.push(extras[i]);
            }

            i += 1;
        }

        n_interval
    }
    
    #[inline(always)]
    fn diff_comp(
        &self,
        graph_obs: &mut BinaryWriterBuilder,
        curr_node: usize,  
        reference: usize,
        ref_list: &[usize],
        curr_list: &[usize]
    ) -> Result<usize, String> {
        let curr_len = curr_list.len();
        let mut ref_len = ref_list.len();
        
        self.compression_vectors.blocks.borrow_mut().clear();
        self.compression_vectors.extras.borrow_mut().clear();
        self.compression_vectors.left.borrow_mut().clear();
        self.compression_vectors.len.borrow_mut().clear();
        self.compression_vectors.residuals.borrow_mut().clear();

        // let written_data_at_start = graph_obs.len();
        let written_data_at_start = graph_obs.written_bits;

        let mut _t;
        let mut j = 0; // index of the next successor of the current node we must examine
        let mut k = 0; // index of the next successor of the reference node we must examine
        let mut prev = 0;
        let mut curr_block_len = 0; //number of entries (in the reference list) we have already copied/ignored (in the current block)

        let mut copying = true; // true iff we are producing a copy block (instead of an ignore block)

        if reference == 0 {
            ref_len = 0;
        }

        while j < curr_len && k < ref_len {
            if copying { // First case: we are currently copying entries from the reference list
                match curr_list[j].cmp(&ref_list[k]) {
                    Ordering::Greater => {
                        // If while copying we go beyond the current element of the ref list, then we must stop
                        self.compression_vectors.blocks.borrow_mut().push(curr_block_len);
                        copying = false;
                        curr_block_len = 0; 
                    },
                    Ordering::Less => {
                        /* If while copying we find a non-matching element of the reference list which is 
                        larger than us, then we can just add the current element to the extra list and move on,
                        increasing j.
                        */
                        self.compression_vectors.extras.borrow_mut().push(curr_list[j]);
                        j += 1;
                    },
                    Ordering::Equal => {
                        /* If the current elements of the two lists are equal, we increase the block len,
                        increasing both j and k */
                        j += 1;
                        k += 1;
                        curr_block_len += 1;
                    }
                }
            } else if curr_list[j] < ref_list[k] { /* If we did not go beyond the current element of the ref list, 
                we just add the current element to the extra list and move on, increasing j */
                self.compression_vectors.extras.borrow_mut().push(curr_list[j]);
                j += 1;
            } else if curr_list[j] > ref_list[k] { /* If we went beyond the current elem of the reference list,
                we increase the block len and k */
                k += 1;
                curr_block_len += 1;
            } else { /* If we found a match, we flush the current block and start a new copying phase */
                self.compression_vectors.blocks.borrow_mut().push(curr_block_len);
                copying = true;
                curr_block_len = 0;
            }
        }

        /* We only enqueue the last block's len when we were copying 
        and did not copy up to the end of the ref list */
        if copying && k < ref_len {
            self.compression_vectors.blocks.borrow_mut().push(curr_block_len);
        }

        // If there are still missing elements add them to the extra list
        while j < curr_len {
            self.compression_vectors.extras.borrow_mut().push(curr_list[j]);
            j += 1;
        }

        let block_count = self.compression_vectors.blocks.borrow().len();
        let extra_count = self.compression_vectors.extras.borrow().len();

        // If we have a nontrivial reference window we write the reference to the reference list
        if self.window_size > 0 {
            _t = self.write_reference(graph_obs, reference)?;
        }

        // Then, if the reference is not void we write the length of the copy list
        if reference != 0 {
            _t = self.write_block_count(graph_obs, block_count)?;

            // Then, we write the copy list; all lengths except the first one are decremented
            if block_count > 0 {
                _t = self.write_block(graph_obs, self.compression_vectors.blocks.borrow()[0])?;
                for blk in self.compression_vectors.blocks.borrow().iter().skip(1) {
                    _t = self.write_block(graph_obs, blk - 1)?;
                }
            }
        }

        // Finally, we write the extra list
        if extra_count > 0 {
            let residual;
            let residual_count;

            if self.min_interval_len != 0 {
                // If we are to produce intervals, we first compute them
                let interval_count = self.intervalize(
                    &self.compression_vectors.extras.borrow(), 
                    &mut self.compression_vectors.left.borrow_mut(), 
                    &mut self.compression_vectors.len.borrow_mut(), 
                    &mut self.compression_vectors.residuals.borrow_mut()
                );

                _t = GammaCode::write_next(graph_obs, interval_count as u64, self.zeta_k) as usize;

                let mut curr_int_len;

                for i in 0..interval_count {
                    if i == 0 {
                        prev = self.compression_vectors.left.borrow()[i];
                        _t = OutIntervalCoding::write_next(graph_obs, int2nat(prev as i64 - curr_node as i64), self.zeta_k) as usize;
                    } else {
                        _t = OutIntervalCoding::write_next(graph_obs, (self.compression_vectors.left.borrow()[i] - prev - 1) as u64, self.zeta_k) as usize;
                    }
                    
                    curr_int_len = self.compression_vectors.len.borrow()[i];
                    
                    prev = self.compression_vectors.left.borrow()[i] + curr_int_len;
                    
                    _t = OutIntervalCoding::write_next(graph_obs, (curr_int_len - self.min_interval_len) as u64, self.zeta_k) as usize;
                }
                
                residual_count = self.compression_vectors.residuals.borrow().len();
                residual = self.compression_vectors.residuals.borrow();
            } else {
                residual_count = self.compression_vectors.extras.borrow().len();
                residual = self.compression_vectors.extras.borrow();
            }

            // Now we write out the residuals, if any
            if residual_count != 0 {
                prev = residual[0];
                _t = self.write_residual(graph_obs, int2nat(prev as i64 - curr_node as i64) as usize)?;
                for i in 1..residual_count {
                    if residual[i] == prev {
                        return Err(format!("Repeated successor {} in successor list of node {}", prev, curr_node));
                    }
                    
                    _t = self.write_residual(graph_obs, residual[i] - prev - 1)?;
                    prev = residual[i];
                }
            }
        }

        Ok(graph_obs.written_bits /* graph_obs.len() */ - written_data_at_start)
    }

    #[inline(always)]
    fn write_reference(&self, graph_obs: &mut BinaryWriterBuilder, reference: usize) -> Result<usize, String> {
        if reference > self.window_size {
            return Err("The required reference is incompatible with the window size".to_string());
        }

        OutReferenceCoding::write_next(graph_obs, reference as u64, self.zeta_k);
        Ok(reference)
    }

    #[inline(always)]
    fn write_outdegree(&self, graph_obs: &mut BinaryWriterBuilder, outdegree: usize) -> Result<usize, String> {
        OutOutdegreeCoding::write_next(graph_obs, outdegree as u64, self.zeta_k);
        Ok(outdegree)
    }

    #[inline(always)]
    fn write_block_count(&self, graph_obs: &mut BinaryWriterBuilder, block_count: usize) -> Result<usize, String> {
        OutBlockCountCoding::write_next(graph_obs, block_count as u64, self.zeta_k);
        Ok(block_count)
    }

    #[inline(always)]
    fn write_block(&self, graph_obs: &mut BinaryWriterBuilder, block: usize) -> Result<usize, String> {
        OutBlockCoding::write_next(graph_obs, block as u64, self.zeta_k);
        Ok(block)
    }

    #[inline(always)]
    fn write_residual(&self, graph_obs: &mut BinaryWriterBuilder, residual: usize) -> Result<usize, String> {
        OutResidualCoding::write_next(graph_obs, residual as u64, self.zeta_k);
        Ok(residual)
    }

    #[inline(always)]
    fn write_offset(&self, offset_obs: &mut BinaryWriterBuilder, offset: usize) -> Result<usize, String> {
        OutOffsetCoding::write_next(offset_obs, offset as u64, self.zeta_k);
        Ok(offset)
    }
}

pub struct BVGraphBuilder<
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: Option<u64>,
    bits_outdegrees_codes: BitsLen,
    bits_blocks_codes: BitsLen,
    bits_residuals_codes: BitsLen,
    bits_intervals_left_codes: BitsLen,
    bits_intervals_len_codes: BitsLen,
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
            max_ref_count: 0, 
            window_size: 0, 
            min_interval_len: 0,
            zeta_k: None,
            bits_outdegrees_codes: BitsLen::default(),
            bits_blocks_codes: BitsLen::default(),
            bits_residuals_codes: BitsLen::default(),
            bits_intervals_left_codes: BitsLen::default(),
            bits_intervals_len_codes: BitsLen::default(),
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
    InBlockCoding: Huffman,
    InBlockCountCoding: UniversalCode,
    InOutdegreeCoding: Huffman,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: Huffman,
    InResidualCoding: Huffman,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutdegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
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
            curr += InOffsetCoding::read_next(&mut offsets_ibs, self.zeta_k);
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

    /// Sets the maximum reference chain length.
    /// 
    /// # Arguments
    /// 
    /// * `ref_count` - The maximum length of the chain.
    pub fn set_max_ref_count(mut self, ref_count: usize) -> Self {
        self.max_ref_count = ref_count;

        self
    }

    /// Sets the maximum reference window size.
    /// 
    /// # Arguments
    /// 
    /// * `window_size` - The maximum length of the window.
    pub fn set_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;

        self
    }

    /// Sets the minimum length of the intervals.
    /// 
    /// # Arguments
    /// 
    /// * `min_interval_length` - The minimum length of the intervals.
    pub fn set_min_interval_len(mut self, min_interval_len: usize) -> Self {
        self.min_interval_len = min_interval_len;

        self
    }

    // Sets the `k` parameter for *zeta*-coding, if present.
    /// 
    /// # Arguments
    /// 
    /// * `zk` - An option containing the value of *k*. If it is not `None` its value has to be >= 1.
    pub fn set_zeta(mut self, zk: Option<u64>) -> Self {
        self.zeta_k = zk;

        self
    }

    // Sets the bits length parameters for the outdegree's Huffman structure.
    /// 
    /// # Arguments
    /// 
    /// * `bits_params` - An instance of `BitsLen` containing two parameters representing the maximum number of Huffman's code
    /// bits and the number of bits that represent the number of values, respectively.
    pub fn set_huff_outdegrees_parameters(mut self, bits_params: BitsLen) -> Self {
        self.bits_outdegrees_codes = bits_params;

        self
    }

    // Sets the bits length parameters for the block's Huffman structure.
    /// 
    /// # Arguments
    /// 
    /// * `bits_params` - An instance of `BitsLen` containing two parameters representing the maximum number of Huffman's code
    /// bits and the number of bits that represent the number of values, respectively.
    pub fn set_huff_blocks_parameters(mut self, bits_params: BitsLen) -> Self {
        self.bits_blocks_codes = bits_params;

        self
    }

    // Sets the bits length parameters for the residual's Huffman structure.
    /// 
    /// # Arguments
    /// 
    /// * `bits_params` - An instance of `BitsLen` containing two parameters representing the maximum number of Huffman's code
    /// bits and the number of bits that represent the number of values, respectively.
    pub fn set_huff_residuals_parameters(mut self, bits_params: BitsLen) -> Self {
        self.bits_residuals_codes = bits_params;

        self
    }

    // Sets the bits length parameters for the left interval's Huffman structure.
    /// 
    /// # Arguments
    /// 
    /// * `bits_params` - An instance of `BitsLen` containing two parameters representing the maximum number of Huffman's code
    /// bits and the number of bits that represent the number of values, respectively.
    pub fn set_huff_intervals_left_parameters(mut self, bits_params: BitsLen) -> Self {
        self.bits_intervals_left_codes = bits_params;

        self
    }

    // Sets the bits length parameters for the length interval's Huffman structure.
    /// 
    /// # Arguments
    /// 
    /// * `bits_params` - An instance of `BitsLen` containing two parameters representing the maximum number of Huffman's code
    /// bits and the number of bits that represent the number of values, respectively.
    pub fn set_huff_intervals_len_parameters(mut self, bits_params: BitsLen) -> Self {
        self.bits_intervals_len_codes = bits_params;

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
            graph_binary_wrapper: Rc::new(RefCell::new(self.graph_binary_wrapper)),
            outdegrees_binary_wrapper: RefCell::new(self.outdegrees_binary_wrapper),
            cached_node: Cell::new(self.cached_node), 
            cached_outdegree: Cell::new(self.cached_outdegree), 
            cached_ptr: Cell::new(self.cached_ptr), 
            max_ref_count: self.max_ref_count, 
            window_size: self.window_size,
            min_interval_len: self.min_interval_len,
            zeta_k: self.zeta_k,
            compression_vectors: CompressionVectors::default(),
            decompression_stats: RefCell::new(DecompressionStats::default()),
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