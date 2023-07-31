use std::{fs, vec, cmp::Ordering, marker::PhantomData, str::FromStr, fmt::Display};

use num_traits::ToPrimitive;
use serde::{Serialize, Deserialize};
use sucds::{mii_sequences::{EliasFanoBuilder, EliasFano}, Serializable};

use crate::{ImmutableGraph, Properties, int2nat, EncodingType, nat2int};
use crate::uncompressed_graph::UncompressedGraph;
use crate::bitstreams::{BinaryReader, BinaryWriterBuilder};

pub trait UniversalCode {
    fn read_next(reader: &mut BinaryReader, zk: Option<u64>) -> u64;
    fn write_next(writer: &mut BinaryWriterBuilder, x: u64, zk: Option<u64>) -> u64;
    fn to_encoding_type() -> EncodingType;
}

pub struct UnaryCode;

impl UniversalCode for UnaryCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, _zk: Option<u64>) -> u64 {
        assert!(reader.fill < 64);

        if reader.fill < 16 {
            reader.refill();
        }

        let mut x = u32::leading_zeros((reader.current as u32) << (32 - reader.fill));
        if x < reader.fill as u32{
            reader.read_bits += x as usize + 1;
            reader.fill -= x as usize + 1;
            return x as u64;
        }

        x = reader.fill as u32;
        let mut read = reader.read();

        if read.is_ok() {
            reader.current = read.unwrap();
            while reader.current == 0 && read.is_ok() {
                x += 8;
                read = reader.read();
                if let Ok(r) = read {
                    reader.current = r;
                }
            }
        }

        reader.fill = (63 - u64::leading_zeros(reader.current)) as usize;
        x += 7 - reader.fill as u32;
        reader.read_bits += x as usize + 1;
        x as u64
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriterBuilder, x: u64, _zk: Option<u64>) -> u64 {
        if x < writer.free as u64 {
            return writer.write_in_current(1, x + 1);
        }

        let shift = writer.free;
        let x = x - shift as u64;

        writer.written_bits += shift;
        writer.write(writer.current);
        writer.free = 8;
        writer.current = 0;
        
        let mut i = x >> 3;

        writer.written_bits += x as usize & 0x7FFFFFF8;
        
        while i != 0 {
            writer.write(0);
            i -= 1;
        }

        writer.write_in_current(1, (x & 7) + 1);

        x + shift as u64 + 1
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::UNARY
    }
}

pub struct GammaCode;

impl UniversalCode for GammaCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, _zk: Option<u64>) -> u64 {
        let msb = UnaryCode::read_next(reader, None);
        ((1 << msb) | reader.read_int(msb)) - 1
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriterBuilder, x: u64, _zk: Option<u64>) -> u64 {
        assert!(x < u64::MAX);
        // if x < MAX_PRECOMPUTED TODO

        let x = x + 1; // Code [0, +inf - 1]
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;

        UnaryCode::write_next(writer, msb, None) + writer.push_bits(x, msb)
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::GAMMA
    }
}

pub struct DeltaCode;

impl UniversalCode for DeltaCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, _zk: Option<u64>) -> u64 {
        let msb = GammaCode::read_next(reader, None);
        ((1 << msb) | reader.read_int(msb)) - 1
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriterBuilder, x: u64, _zk: Option<u64>) -> u64 {
        assert!(x < u64::MAX);
        // if x < MAX_PRECOMPUTED TODO

        let x =  x + 1; // Code [0, +inf - 1]
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
        GammaCode::write_next(writer, msb, None) + writer.push_bits(x, msb)
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::DELTA
    }
}

pub struct ZetaCode;

impl UniversalCode for ZetaCode {
    #[inline(always)]
    fn read_next(reader: &mut BinaryReader, zk: Option<u64>) -> u64 {
        let zk = zk.unwrap();
        assert!(zk >= 1);

        let unary = UnaryCode::read_next(reader, None);
        let left = 1 << (unary * zk);
        let m = reader.read_int(unary * zk + zk - 1);
        if m < left {m + left - 1} else {(m << 1) + reader.read_from_current(1) - 1}
    }

    #[inline(always)]
    fn write_next(writer: &mut BinaryWriterBuilder, x: u64, zk: Option<u64>) -> u64 {
        let zk = zk.unwrap();
        assert!(x < u64::MAX);
        assert!(zk < u64::MAX);

        let x = x + 1;
        let msb = (u64::BITS - 1 - x.leading_zeros()) as u64;
        let h = msb / zk;
        let unary = UnaryCode::write_next(writer, h, None);
        let left = 1 << (h * zk);
        unary + 
            if x - left < left 
                {writer.push_bits(x - left, h * zk + zk - 1)}
            else 
                {writer.push_bits(x, h * zk + zk)}
    }

    #[inline(always)]
    fn to_encoding_type() -> EncodingType {
        EncodingType::ZETA
    }
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraph<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    n: usize,
    m: usize,
    pub graph_memory: Box<[T]>,
    pub offsets: Vec<usize>,  // TODO: it is converted from an EliasFanoLongMonotoneList
    cached_node: Option<usize>,
    cached_outdegree: Option<usize>,
    cached_ptr: Option<usize>,
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: Option<u64>,
    _phantom_block_coding: PhantomData<BlockCoding>,
    _phantom_block_count_coding: PhantomData<BlockCountCoding>,
    _phantom_outdegree_coding: PhantomData<OutdegreeCoding>,
    _phantom_offset_coding: PhantomData<OffsetCoding>,
    _phantom_reference_coding: PhantomData<ReferenceCoding>,
    _phantom_residual_coding: PhantomData<ResidualCoding>,
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> ImmutableGraph for BVGraph<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize> 
        + Display
        + Copy
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
        if self.cached_node.is_some() && x == self.cached_node.unwrap() {
            return self.cached_outdegree;
        }
        
        if x >= self.n {
            return None;
        }

        self.cached_node = Some(x);

        let mut node_iter = BVGraphIterator {
            curr: 0, 
            graph: self.as_ref(), 
            _phantom_block_coding: PhantomData, 
            _phantom_block_count_coding: PhantomData, 
            _phantom_outdegree_coding: PhantomData, 
            _phantom_offset_coding: PhantomData, 
            _phantom_reference_coding: PhantomData, 
            _phantom_residual_coding: PhantomData, 
            _phantom_t: PhantomData 
        };
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1]}).ok()?;

        self.cached_outdegree = 
            if let Some(outd) = self.read_outdegree(&mut node_iter) {
                outd.to_usize()
            } else {
                None
            };

        self.cached_outdegree
    }

    fn store(&self, filename: &str) -> std::io::Result<()> {
        let mut bit_offset = 0;

        // let mut bit_count = Vec::default();
        let mut bit_count = BinaryWriterBuilder::new();

        // let mut graph_buf = Vec::default();
        // let mut offsets_buf = Vec::default();

        let mut graph_obs = BinaryWriterBuilder::new();
        let mut offsets_obs = BinaryWriterBuilder::new();

        let cyclic_buff_size = self.window_size + 1;
        // Cyclic array of previous lists
        let mut list = vec![vec![0; 1024]; cyclic_buff_size];
        // The length of each list
        let mut list_len = vec![0; cyclic_buff_size];
        // The depth of the references of each list
        let mut ref_count: Vec<i32> = vec![0; cyclic_buff_size];

        let mut node_iter = BVGraphIterator {
            curr: 0, 
            graph: self.as_ref(), 
            _phantom_block_coding: PhantomData, 
            _phantom_block_count_coding: PhantomData, 
            _phantom_outdegree_coding: PhantomData, 
            _phantom_offset_coding: PhantomData, 
            _phantom_reference_coding: PhantomData, 
            _phantom_residual_coding: PhantomData, 
            _phantom_t: PhantomData 
        };
        
        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.next().unwrap().to_usize().unwrap();
            let curr_idx = curr_node.to_usize().unwrap() % cyclic_buff_size;
            
            // println!("Curr node: {}", curr_node);
            
            // Doesn't use delta (graph_buf.len() - bit_offset) since it has to be monotonically increasing for EliasFano
            // self.write_offset(&mut offsets_buf, graph_buf.len()).unwrap();
            
            self.write_offset(&mut offsets_obs, graph_obs.written_bits - bit_offset).unwrap();
            
            // bit_offset = graph_buf.len();

            bit_offset = graph_obs.written_bits;

            // self.write_outdegree(&mut graph_buf, outd).unwrap();
            
            self.write_outdegree(&mut graph_obs, outd).unwrap();

            if outd > list[curr_idx].len() {
                list[curr_idx].resize(outd, 0);
            }
            
            let mut successors = Vec::default();
            let mut successors_it = self.successors_plain(curr_node.to_usize().unwrap()).unwrap();

            while successors_it.has_next() {
                successors.push(successors_it.next().unwrap().to_usize().unwrap());
            }
            
            list[curr_idx] = successors;
            list_len[curr_idx] = outd;
            
            
            if outd > 0 {
                let mut best_comp = i64::MAX;
                let mut best_cand = -1;
                let mut best_ref: i32 = -1;
                let mut cand;
                
                ref_count[curr_idx] = -1;

                for r in 0..cyclic_buff_size {
                    cand = ((curr_node.to_usize().unwrap() + cyclic_buff_size - r) % cyclic_buff_size) as i32;
                    if ref_count[cand as usize] < (self.max_ref_count as i32) && list_len[cand as usize] != 0 {
                        let diff_comp = 
                            self.diff_comp(&mut bit_count, 
                                            curr_node.to_usize().unwrap(), 
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
                                    
                assert!(best_cand >= 0);
                
                ref_count[curr_idx] = ref_count[best_cand as usize] + 1;
                self.diff_comp(
                    &mut graph_obs, 
                    curr_node.to_usize().unwrap(), 
                    best_ref as usize, 
                    list[best_cand as usize].as_slice(), 
                    list[curr_idx].as_slice(),
                ).unwrap(); // TODO: manage?
                // self.diff_comp(
                //     &mut graph_buf,
                //     curr_node,
                //     best_ref as usize,
                //     list[best_cand as usize].as_slice(), 
                //     list[curr_idx].as_slice()
                // ).unwrap();
            }
            
            node_iter.advance_by(outd).unwrap();
        }
        
        // We write the final offset to the offset stream
        // Doesn't use delta (graph_buf.len() - bit_offset) since it has to be monotonically increasing for EliasFano
        // self.write_offset(&mut offsets_buf, graph_buf.len()).unwrap();
        
        self.write_offset(&mut offsets_obs, graph_obs.written_bits - bit_offset).unwrap(); // TODO: manage?
        
        // let universe = *offsets_buf.last().unwrap() + 1;
        // let num_vals = offsets_buf.len();
        // let mut efb = EliasFanoBuilder::new(universe, num_vals).unwrap(); // TODO: check parameters

        // for offset in offsets_buf.iter() {
        //     efb.push(*offset).unwrap();
        // }

        // let ef = efb.build();

        // let mut bytes = Vec::default();
        // ef.serialize_into(&mut bytes).unwrap();
        // fs::write(format!("{}.offsets", filename), bytes)?;
        
        // let graph_buf: Vec<String> = graph_buf.into_iter().map(|val| format!("{} ", val)).collect();
        // let graph_buf = graph_buf.concat();
        
        // fs::write(format!("{}.graph", filename), graph_buf)?;

        ///////////////////////////////////////////////////////////////////////

        let graph = graph_obs.build();
        let offsets = offsets_obs.build();

        fs::write(format!("{}.offsets", filename), bincode::serialize(&offsets.os).unwrap()).unwrap();
        fs::write(format!("{}.graph", filename), bincode::serialize(&graph.os).unwrap()).unwrap();

        let props = Properties {
            nodes: self.n,
            arcs: self.m,
            window_size: self.window_size,
            max_ref_count: self.max_ref_count,
            min_interval_len: self.min_interval_len,
            zeta_k: self.zeta_k,
            outdegree_coding: OutdegreeCoding::to_encoding_type(),
            block_coding: BlockCoding::to_encoding_type(),
            residual_coding: ResidualCoding::to_encoding_type(),
            reference_coding: ReferenceCoding::to_encoding_type(),
            block_count_coding: BlockCountCoding::to_encoding_type(),
            offset_coding: OffsetCoding::to_encoding_type(),
            avg_ref: 0.,
            avg_dist: 0.,
            copied_arcs: 0,
            intervalized_arcs: 0,
            residual_arcs: 0,
            bits_per_link: 0.,
            comp_ratio: 0.,
            bits_per_node: 0.,
            avg_bits_for_outdeg: 0.,
            avg_bits_for_refs: 0.,
            avg_bits_for_blocks: 0.,
            avg_bits_for_residuals: 0.,
            avg_bits_for_intervals: 0.,
            bits_for_outdeg: 0,
            bits_for_refs: 0,
            bits_for_blocks: 0,
            bits_for_residuals: 0,
            bits_for_intervals: 0
        };

        fs::write(format!("{}.properties", filename), serde_json::to_string(&props).unwrap())?;

        Ok(())
    }
}

pub struct BVGraphNodeIterator<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> 
where T:
    num_traits::Num 
    + PartialOrd 
    + num_traits::ToPrimitive
    + serde::Serialize
    + Clone
    + From<usize>
    + Display
    + Copy
{
    // The number of nodes
    n: usize,
    // The graph on which we iterate
    graph: BV,
    // The input bit stream
    ibs: BinaryReader,
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
    _phantom_block_coding: PhantomData<BlockCoding>,
    _phantom_block_count_coding: PhantomData<BlockCountCoding>,
    _phantom_outdegree_coding: PhantomData<OutdegreeCoding>,
    _phantom_offset_coding: PhantomData<OffsetCoding>,
    _phantom_reference_coding: PhantomData<ReferenceCoding>,
    _phantom_residual_coding: PhantomData<ResidualCoding>,
    _phantom_t: PhantomData<T>,
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> Iterator for BVGraphNodeIterator<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T,
        BV
> 
where T:
    num_traits::Num 
    + PartialOrd 
    + num_traits::ToPrimitive
    + serde::Serialize
    + Clone
    + From<usize>
    + Display
    + Copy
{
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
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> BVGraphNodeIterator<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T,
        BV
> 
where T:
    num_traits::Num 
    + PartialOrd 
    + num_traits::ToPrimitive
    + serde::Serialize
    + Clone
    + From<usize>
    + Display
    + Copy
{
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

/// Defines an iterator over all the elements of the graph vector.
pub struct BVGraphIterator<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    curr: usize,
    graph: BV,
    _phantom_block_coding: PhantomData<BlockCoding>,
    _phantom_block_count_coding: PhantomData<BlockCountCoding>,
    _phantom_outdegree_coding: PhantomData<OutdegreeCoding>,
    _phantom_offset_coding: PhantomData<OffsetCoding>,
    _phantom_reference_coding: PhantomData<ReferenceCoding>,
    _phantom_residual_coding: PhantomData<ResidualCoding>,
    _phantom_t: PhantomData<T>,
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> Iterator for BVGraphIterator<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T,
    BV
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr > self.graph.as_ref().graph_memory.len() {
            return None;
        }

        let res = Some(self.graph.as_ref().graph_memory[self.curr]);

        self.curr += 1;

        res
    }
    // fn outdegrees(&self) -> iter;
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> BVGraphIterator<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T,
        BV
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    /// Positions the iterator to the given index (of the graph vector), without consuming the element.
    /// 
    /// Returns [`Ok(())`][Ok] if the positioning operation was successful, or an error otherwise.
    /// 
    /// # Arguments
    /// 
    /// * `n` - The position to move the iterator to.
    /// 
    /// # Examples
    /// 
    /// ```
    /// graph.iter().position_to(7);  // Where graph is a BVGraph with >7 elements in its graph vector
    /// ```
    fn position_to(&mut self, n: usize) -> Result<(), &str> {
        if n > self.graph.as_ref().graph_memory.len() - 1 {
            return Err("The provided position exceeds the number of possible positions in the graph vector");
        }

        if n < self.curr {
            return Err("The provided position comes before the current position of the iterator");
        }

        while self.curr <= n {
            self.curr += 1;
        }

        Ok(())
    }

    /// Advances the iterator by a certain amount.
    /// 
    /// Returns [`Ok(())`][Ok] if the advancing operation was successful, or an error otherwise.
    /// 
    /// # Arguments
    /// 
    /// * `n` - The number of positions to advance.
    /// 
    /// # Examples
    /// 
    /// ```
    /// let it = graph.iter();
    /// it.next();  // First element
    /// it.next();  // Second element
    /// graph.iter().advance_by(3);
    /// it.next();  // Fifth element
    /// ```
    fn advance_by(&mut self, n: usize) -> Result<(), &str> {
        if n > self.graph.as_ref().graph_memory.len() - 1 {
            return Err("The provided position exceeds the number of possible positions in the graph vector");
        }

        let target = self.curr + n;

        while self.curr < target {
            self.curr += 1;
        }

        Ok(())
    }

    /// Returns `true` if the iterator has not reached the final node of the graph, `false` otherwise.
    #[inline]
    fn has_next(&self) -> bool {
        self.curr < self.graph.as_ref().graph_memory.len()
    }
}

/// Defines an iterator over the successors of a node in a graph.
#[derive(Debug)]
pub struct BVGraphSuccessorsIterator<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    base: usize,
    idx_from_base: usize,
    up_to: usize,
    graph: BV,
    _phantom_block_coding: PhantomData<BlockCoding>,
    _phantom_block_count_coding: PhantomData<BlockCountCoding>,
    _phantom_outdegree_coding: PhantomData<OutdegreeCoding>,
    _phantom_offset_coding: PhantomData<OffsetCoding>,
    _phantom_reference_coding: PhantomData<ReferenceCoding>,
    _phantom_residual_coding: PhantomData<ResidualCoding>,
    _phantom_t: PhantomData<T>
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> Iterator for BVGraphSuccessorsIterator<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T,
    BV
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() {
            return None;
        }

        let g = self.graph.as_ref();
        
        self.idx_from_base += 1;        
        
        Some(g.graph_memory[self.base + self.idx_from_base])
    }
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T,
    BV: AsRef<BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
>>> BVGraphSuccessorsIterator<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T,
    BV
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    /// Returns `true` if the iterator has not reached the final successor of the node, `false` otherwise.
    #[inline(always)]
    fn has_next(&self) -> bool {
        self.base + self.idx_from_base + 1 < self.up_to
    }
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> AsMut<BVGraph<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
>> for BVGraph<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{

    fn as_mut(&mut self) -> &mut BVGraph<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
    > {
        self
    }
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> AsRef<BVGraph<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
>> for BVGraph<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    fn as_ref(&self) -> &BVGraph<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding, T> {
        self
    }
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> IntoIterator for BVGraph<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    type Item = T;

    type IntoIter = BVGraphIterator<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T,
        BVGraph<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding, T>>;

    fn into_iter(self) -> Self::IntoIter {
        BVGraphIterator {
            curr: 0,
            graph: self,
            _phantom_block_coding: PhantomData,
            _phantom_block_count_coding: PhantomData,
            _phantom_outdegree_coding: PhantomData,
            _phantom_offset_coding: PhantomData,
            _phantom_reference_coding: PhantomData,
            _phantom_residual_coding: PhantomData,
            _phantom_t: PhantomData
        }
    }
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> BVGraph<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + Display
        + Copy
{
    pub fn iter(&self) -> BVGraphNodeIterator<
        BlockCoding, 
        BlockCountCoding, 
        OutdegreeCoding, 
        OffsetCoding, 
        ReferenceCoding, 
        ResidualCoding,
        T,
        &Self
    > {
        BVGraphNodeIterator {
            n: self.n,
            graph: self,
            ibs: BinaryReader::new(self.graph_memory.iter().map(|x| x.to_u8().unwrap()).collect()),
            cyclic_buffer_size: self.window_size + 1,
            window: vec![vec![0usize; self.window_size + 1]; 1024],
            outd: vec![0usize; self.window_size + 1],
            from: 0,
            curr: -1,
            _phantom_block_coding: PhantomData,
            _phantom_block_count_coding: PhantomData,
            _phantom_offset_coding: PhantomData,
            _phantom_outdegree_coding: PhantomData,
            _phantom_reference_coding: PhantomData,
            _phantom_residual_coding: PhantomData,
            _phantom_t: PhantomData
        }
    }

    fn outdegree_internal(&self, x: usize, decoder: &mut BinaryReader) -> usize { // TODO: reintroduce mut and caches
        if self.cached_node.is_some() && x == self.cached_node.unwrap() {
            return self.cached_outdegree.unwrap();
        }
        
        println!("offset of x {}", self.offsets[x]);
        decoder.position(self.offsets[x] as u64); // TODO: offsets are encoded
        // self.cached_node = Some(x);
        let d = OutdegreeCoding::read_next(decoder, self.zeta_k) as usize;
        println!("found outdeg {}", d);
        // self.cached_outdegree = Some(d);
        // self.cached_ptr = Some(decoder.position);
        d
    }

    fn decode_list(&self, x: usize, decoder: &mut BinaryReader, window: Option<&mut Vec<Vec<usize>>>, outd: &mut [usize]) -> Box<[usize]> {
        println!("passed node {}", x);
        
        let cyclic_buffer_size = self.window_size + 1;

        let degree;
        if window.is_none() {
            println!("window is none");
            degree = self.outdegree_internal(x, decoder);
            decoder.position(degree as u64);
        } else {
            degree = OutdegreeCoding::read_next(decoder, self.zeta_k) as usize;
            outd[x % cyclic_buffer_size] = degree; 
        }

        println!("decoded a degree of {}", degree);

        if degree == 0 {
            return Box::new([]);
        }

        let mut reference = -1;
        if self.window_size > 0 {
            reference = ReferenceCoding::read_next(decoder, self.zeta_k) as i64;
        }

        // Position in the circular buffer of the reference of the current node
        let reference_index = ((x as i64 - reference + cyclic_buffer_size as i64) as usize) % cyclic_buffer_size;
        println!("computed reference_index {}", reference_index);

        let mut block = Vec::default();

        let mut extra_count;

        if reference > 0 {            
            let block_count = BlockCountCoding::read_next(decoder, self.zeta_k) as usize; 
            println!("read block_count {}", block_count);
            if block_count != 0 {
                block = Vec::with_capacity(block_count);
            }

            let mut copied = 0; // # of copied successors
            let mut total = 0; // total # of successors specified in some copy block

            let mut i = 0;
            while i < block_count {
                block.push(BlockCoding::read_next(decoder, self.zeta_k) as usize + if i == 0 {0} else {1});
                total += block[i];
                if (i & 1) == 0 { // Alternate, count only even blocks
                    copied += block[i];
                }

                i += 1;
            }

            // If the block count is even, we must comput the number of successors copied implicitly
            if (block_count & 1) == 0 {
                copied += (if window.is_some() {outd[reference_index]} else {self.outdegree_internal((x as i64 - reference) as usize, decoder)}) - total;
            }
            println!("degree {}, copied {}", degree, copied);
            extra_count = degree - copied;
        } else {
            extra_count = degree;
        }

        let mut interval_count = 0; // Number of intervals

        let mut left = Vec::default();
        let mut len = Vec::default();

        if extra_count > 0 && self.min_interval_len != 0 {
            interval_count = GammaCode::read_next(decoder, self.zeta_k) as usize;
            
            if interval_count != 0 {
                let mut prev; // Holds the last integer in the last interval
                left = Vec::with_capacity(interval_count);
                len = Vec::with_capacity(interval_count);

                left.push(nat2int(GammaCode::read_next(decoder, self.zeta_k)) + x as i64);
                len.push(GammaCode::read_next(decoder, self.zeta_k) as usize + self.min_interval_len);
                
                println!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa {}, {:?}", extra_count, len);
                prev = left[0] + len[0] as i64;
                extra_count -= len[0];

                let mut i = 1;
                while i < interval_count {
                    prev += GammaCode::read_next(decoder, self.zeta_k) as i64 + 1;

                    left.push(prev);
                    len.push(GammaCode::read_next(decoder, self.zeta_k) as usize + self.min_interval_len);

                    prev += len[i] as i64;
                    extra_count -= len[i];

                    i += 1;
                }
            }
        }

        let mut residual_list = Vec::with_capacity(extra_count);
        if extra_count > 0 {
            residual_list.push(x as i64 + nat2int(ResidualCoding::read_next(decoder, self.zeta_k)));
            let mut remaining = extra_count - 1;
            let mut curr_len = 1;

            while remaining > 0 {
                residual_list.push(residual_list[curr_len - 1] + ResidualCoding::read_next(decoder, self.zeta_k) as i64 + 1);
                curr_len += 1;

                remaining -= 1;
            }
        }

        // The extra part is made by the contribution of intervals, if any, and by the residuals list.
        let mut extra_list = if interval_count == 0 {residual_list.clone()} else {Vec::default()};
        if interval_count > 0 {
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
        } 
        if interval_count > 0 && extra_count > 0 {
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

        let mut block_list = Vec::default();
        if reference > 0 {
            let decoded_reference;

            let mut reference_it = 
                if let Some(window) = window {
                    window[reference_index][0..outd[reference_index]].iter()
                } else {
                    decoded_reference = self.decode_list(
                        (x as i64 - reference) as usize, 
                        &mut BinaryReader::new(self.graph_memory.iter().map(|x| x.to_u8().unwrap()).collect()), 
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

                    // let mut i = 0;
                    // while i < block[curr_mask] && reference_it.next().is_some() {
                    //     i += 1;
                    // }
                    // curr_mask += 1;

                    left = if curr_mask < mask_len {curr_mask += 1; block[curr_mask - 1] as i64} else {-1};
                }
            } else {
                left = -1;
            }

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

                        // let mut i = 0;
                        // while i < block[curr_mask] && reference_it.next().is_some() {
                        //     i += 1;
                        // }

                        left = if curr_mask < mask_len {curr_mask += 1; block[curr_mask - 1] as i64} else {-1};
                    }
                    block_list.push(*next.unwrap());
                }
                
            }
        }

        if reference <= 0 {
            let extra_list: Vec<usize> = extra_list.iter().map(|x| x.to_usize().unwrap()).collect();
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
            
            // We write the final offset to the offset            
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
                                    
                assert!(best_cand >= 0);
                
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

    pub fn store2(&mut self, filename: &str) -> std::io::Result<()> {      
        let mut graph_obs = BinaryWriterBuilder::new();
        let mut offsets_obs = BinaryWriterBuilder::new();

        self.compress(&mut graph_obs, &mut offsets_obs);
        
        let graph = graph_obs.build();
        let offsets = offsets_obs.build();
        let props = Properties {
            nodes: self.n,
            arcs: self.m,
            window_size: self.window_size,
            max_ref_count: self.max_ref_count,
            min_interval_len: self.min_interval_len,
            zeta_k: self.zeta_k,
            outdegree_coding: OutdegreeCoding::to_encoding_type(),
            block_coding: BlockCoding::to_encoding_type(),
            residual_coding: ResidualCoding::to_encoding_type(),
            reference_coding: ReferenceCoding::to_encoding_type(),
            block_count_coding: BlockCountCoding::to_encoding_type(),
            offset_coding: OffsetCoding::to_encoding_type(),
            ..Default::default()
        };

        // fs::write(format!("{}.offsets", filename), bincode::serialize(&offsets.os).unwrap()).unwrap();
        // fs::write(format!("{}.graph", filename), bincode::serialize(&graph.os).unwrap()).unwrap();

        fs::write(format!("{}.graph", filename), graph.os).unwrap();
        fs::write(format!("{}.offsets", filename), offsets.os).unwrap();
        fs::write(format!("{}.properties", filename), serde_json::to_string(&props).unwrap())?;

        Ok(())
    }
    
    fn outdegree_internal_plain(&self, x: usize) -> T {
        let mut node_iter = BVGraphIterator {
            curr: 0, 
            graph: self, 
            _phantom_block_coding: PhantomData, 
            _phantom_block_count_coding: PhantomData, 
            _phantom_outdegree_coding: PhantomData, 
            _phantom_offset_coding: PhantomData, 
            _phantom_reference_coding: PhantomData, 
            _phantom_residual_coding: PhantomData, 
            _phantom_t: PhantomData 
        };
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1]}).ok();
        self.read_outdegree(&mut node_iter).unwrap()
    }

    pub fn successors(&self, x: usize) -> Box<[usize]> {
        assert!(x < self.n, "Node index out of range {}", x);
        let mut reader = BinaryReader::new(self.graph_memory.iter().map(|x| x.to_u8().unwrap()).collect());
        self.decode_list(x, &mut reader, None, &mut [])
    }

    
    fn successors_plain(&self, x: usize) -> Option<BVGraphSuccessorsIterator<
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T,
        &BVGraph<
            BlockCoding,
            BlockCountCoding,
            OutdegreeCoding,
            OffsetCoding,
            ReferenceCoding,
            ResidualCoding,
            T
        >>> 
    {
        if x > self.n - 1 {
            return None;
        }

        let base = if x == 0 {0} else { self.offsets[x - 1] };
        Some(BVGraphSuccessorsIterator {
            base,
            idx_from_base: 1, // starts from the outdeg
            up_to: base + self.outdegree_internal_plain(x).to_usize().unwrap() + 2, // summing 2 to skip the node and its outdeg
            graph: self,
            _phantom_block_coding: PhantomData,
            _phantom_block_count_coding: PhantomData,
            _phantom_outdegree_coding: PhantomData,
            _phantom_offset_coding: PhantomData,
            _phantom_reference_coding: PhantomData,
            _phantom_residual_coding: PhantomData,
            _phantom_t: PhantomData
        })
    }

    fn read_outdegree(
        &self, 
        outdegree_iter: &mut BVGraphIterator<
            BlockCoding,
            BlockCountCoding,
            OutdegreeCoding,
            OffsetCoding,
            ReferenceCoding,
            ResidualCoding,
            T,
            &Self
        >
    ) -> Option<T> {
        outdegree_iter.next()
        // TODO: implement outdegree_iter.read_gamma()
        // TODO: implement outdegree_iter.read_delta()
    }

    #[inline(always)]
    fn intervalize(
        &self, 
        v: &Vec<usize>,
        left: &mut Vec<usize>, 
        len: &mut Vec<usize>, 
        residuals: &mut Vec<usize>
    ) -> usize {
        let mut n_interval = 0;
        let v_len = v.len();

        let mut j;

        left.clear();
        len.clear();
        residuals.clear();

        let mut i = 0;

        while i < v_len {
            j = 0;
            if i < v_len - 1 && v[i] + 1 == v[i + 1] {
                j += 1;
                while i + j < v_len - 1 && v[i + j] + 1 == v[i + j + 1] {
                    j += 1;
                }
                j += 1;

                // Now j is the # of integers in the interval
                if j >= self.min_interval_len {
                    left.push(v[i]);
                    len.push(j);
                    n_interval += 1;
                    i += j - 1;
                }
            }

            if j < self.min_interval_len {
                residuals.push(v[i]);
            }

            i += 1;
        }

        n_interval
    }
    
    fn diff_comp(
        &self,
        graph_obs: &mut BinaryWriterBuilder,
        // graph_obs: &mut Vec<usize>,
        curr_node: usize,  
        reference: usize,
        ref_list: &[usize],
        curr_list: &[usize]
    ) -> Result<usize, String> {
        let curr_len = curr_list.len();
        let mut ref_len = ref_list.len();

        // TODO: move out to avoid recreating at each call
        let mut blocks = Vec::<usize>::default();
        let mut extras = Vec::<usize>::default();
        let mut left = Vec::<usize>::default();
        let mut len = Vec::<usize>::default();
        let mut residuals = Vec::<usize>::default();

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
                        blocks.push(curr_block_len);
                        copying = false;
                        curr_block_len = 0; 
                    },
                    Ordering::Less => {
                        /* If while copying we find a non-matching element of the reference list which is 
                        larger than us, then we can just add the current element to the extra list and move on,
                        increasing j.
                        */
                        extras.push(curr_list[j]);
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
                extras.push(curr_list[j]);
                j += 1;
            } else if curr_list[j] > ref_list[k] { /* If we went beyond the current elem of the reference list,
                we increase the block len and k */
                k += 1;
                curr_block_len += 1;
            } else { /* If we found a match, we flush the current block and start a new copying phase */
                blocks.push(curr_block_len);
                copying = true;
                curr_block_len = 0;
            }
        }

        /* We only enqueue the last block's len when we were copying 
        and did not copy up to the end of the ref list */
        if copying && k < ref_len {
            blocks.push(curr_block_len);
        }

        // If there are still missing elements add them to the extra list
        while j < curr_len {
            extras.push(curr_list[j]);
            j += 1;
        }

        let block_count = blocks.len();
        let extra_count = extras.len();

        // If we have a nontrivial reference window we write the reference to the reference list
        if self.window_size > 0 {
            _t = self.write_reference(graph_obs, reference)?;
        }

        // Then, if the reference is not void we write the length of the copy list
        if reference != 0 {
            _t = self.write_block_count(graph_obs, block_count)?;

            // Then, we write the copy list; all lengths except the first one are decremented
            if block_count > 0 {
                _t = self.write_block(graph_obs, blocks[0])?;
                for blk in blocks.iter().skip(1) {
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
                let interval_count = self.intervalize(&extras, &mut left, &mut len, &mut residuals);

                _t = GammaCode::write_next(graph_obs, interval_count as u64, self.zeta_k) as usize;

                let mut curr_int_len;

                for i in 0..interval_count {
                    if i == 0 {
                        prev = left[i];
                        _t = GammaCode::write_next(graph_obs, int2nat(prev as i64 - curr_node as i64), self.zeta_k) as usize;
                    } else {
                        _t = GammaCode::write_next(graph_obs, (left[i] - prev - 1) as u64, self.zeta_k) as usize;
                    }
                    
                    curr_int_len = len[i];
                    
                    prev = left[i] + curr_int_len;
                    
                    _t = GammaCode::write_next(graph_obs, (curr_int_len - self.min_interval_len) as u64, self.zeta_k) as usize;
                }
                
                residual_count = residuals.len();
                residual = residuals;
            } else {
                residual_count = extras.len();
                residual = extras;
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

    fn write_reference(&self, graph_obs: &mut BinaryWriterBuilder, reference: usize) -> Result<usize, String> {
        if reference > self.window_size {
            return Err("The required reference is incompatible with the window size".to_string());
        }

        ReferenceCoding::write_next(graph_obs, reference as u64, self.zeta_k);
        Ok(reference)
    }

    fn write_outdegree(&self, graph_obs: &mut BinaryWriterBuilder, outdegree: usize) -> Result<usize, String> {
        OutdegreeCoding::write_next(graph_obs, outdegree as u64, self.zeta_k);
        Ok(outdegree)
    }

    fn write_block_count(&self, graph_obs: &mut BinaryWriterBuilder, block_count: usize) -> Result<usize, String> {
        BlockCountCoding::write_next(graph_obs, block_count as u64, self.zeta_k);
        Ok(block_count)
    }

    fn write_block(&self, graph_obs: &mut BinaryWriterBuilder, block: usize) -> Result<usize, String> {
        BlockCoding::write_next(graph_obs, block as u64, self.zeta_k);
        Ok(block)
    }

    fn write_residual(&self, graph_obs: &mut BinaryWriterBuilder, residual: usize) -> Result<usize, String> {
        ResidualCoding::write_next(graph_obs, residual as u64, self.zeta_k);
        Ok(residual)
    }

    fn write_offset(&self, offset_obs: &mut BinaryWriterBuilder, offset: usize) -> Result<usize, String> {
        OffsetCoding::write_next(offset_obs, offset as u64, self.zeta_k);
        Ok(offset)
    }

    // pub fn load(filename: &str) -> Result<Self, &str> {
    //     let prop_json = fs::read_to_string(format!("{}.properties", filename)).unwrap(); // TODO: handle

    //     let props: Properties = serde_json::from_str(prop_json.as_str()).unwrap();

    //     let n = props.nodes;
    //     let m = props.arcs;
    //     let window_size = props.window_size;
    //     let max_ref_count = props.max_ref_count;
    //     let min_interval_len = props.min_interval_len;
    //     let zeta_k = props.zeta_k; // TODO: Handle absence?

    //     let read_graph = fs::read_to_string(format!("{}.graph", filename)).unwrap(); // TODO
    //     let graph_memory = read_graph.split(' ').map(|val| val.parse::<usize>().unwrap()).collect();

    //     let bytes = fs::read(format!("{}.offsets", filename)).unwrap();
    //     let offsets = EliasFano::deserialize_from(bytes.as_slice()).unwrap();

    //     let offsets = offsets.iter(0).collect();

    //     // let read_offsets = fs::read_to_string(format!("{}.offsets", filename)).unwrap();
    //     // let offsets = read_offsets.split(' ').map(|val| val.parse::<usize>().unwrap()).collect();

    //     Ok(Self { 
    //         n, 
    //         m, 
    //         graph_memory, 
    //         offsets, 
    //         cached_node: None, 
    //         cached_outdegree: None, 
    //         cached_ptr: None, 
    //         max_ref_count, 
    //         window_size, 
    //         min_interval_len
    //     })
    // }
}

pub struct BVGraphBuilder<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
{
    pub num_nodes: usize,
    pub num_edges: usize,
    pub loaded_graph: Vec<T>,
    pub loaded_offsets: Vec<usize>,
    pub cached_node: Option<usize>,
    pub cached_outdegree: Option<usize>,
    pub cached_ptr: Option<usize>,
    pub max_ref_count: usize,
    pub window_size: usize,
    pub min_interval_len: usize,
    pub zeta_k: Option<u64>, 
    pub outdegree_coding: EncodingType, 
    pub block_coding: EncodingType, 
    pub residual_coding: EncodingType, 
    pub reference_coding: EncodingType, 
    pub block_count_coding: EncodingType, 
    pub offset_coding: EncodingType, 
    pub _phantom_block_coding: PhantomData<BlockCoding>,
    pub _phantom_block_count_coding: PhantomData<BlockCountCoding>,
    pub _phantom_outdegree_coding: PhantomData<OutdegreeCoding>,
    pub _phantom_offset_coding: PhantomData<OffsetCoding>,
    pub _phantom_reference_coding: PhantomData<ReferenceCoding>,
    pub _phantom_residual_coding: PhantomData<ResidualCoding>,
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> From<UncompressedGraph<T>> for BVGraphBuilder<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
>
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
{
    fn from(graph: UncompressedGraph<T>) -> Self {
        let mut graph_with_outdegrees = Vec::with_capacity(graph.graph_memory.len());

        let mut n = 0;

        for (i, x) in graph.graph_memory.iter().enumerate() {
            graph_with_outdegrees.push(x.clone());
            if n == 0 || graph.offsets[n - 1] == i {
                let outd = graph.outdegree_internal(n.into());
                graph_with_outdegrees.push(T::from(outd));
                n += 1;
            }
        }

        let mut new_offsets = Vec::with_capacity(graph.offsets.len());

        let mut to_add = 1;
        for x in graph.offsets.iter() {
            new_offsets.push(x + to_add);
            to_add += 1;
        }

        Self { 
            num_nodes: graph.num_nodes(), 
            num_edges: graph.num_arcs(), 
            loaded_graph: graph_with_outdegrees,
            loaded_offsets: new_offsets, 
            cached_node: None, 
            cached_outdegree: None, 
            cached_ptr: None, 
            max_ref_count: 3, 
            window_size: 7, 
            min_interval_len: 4,
            zeta_k: Some(3), 
            outdegree_coding: EncodingType::GAMMA, 
            block_coding: EncodingType::GAMMA, 
            residual_coding: EncodingType::ZETA,
            reference_coding: EncodingType::GAMMA, 
            block_count_coding: EncodingType::GAMMA, 
            offset_coding: EncodingType::GAMMA,
            _phantom_block_coding: PhantomData,
            _phantom_block_count_coding: PhantomData,
            _phantom_outdegree_coding: PhantomData,
            _phantom_offset_coding: PhantomData,
            _phantom_reference_coding: PhantomData,
            _phantom_residual_coding: PhantomData
        }
    }
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> Default for BVGraphBuilder<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
>
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
{
    fn default() -> Self {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Vec::default(), 
            loaded_offsets: Vec::default(), 
            cached_node: None, 
            cached_outdegree: None, 
            cached_ptr: None, 
            max_ref_count: 0, 
            window_size: 0, 
            min_interval_len: 0,
            zeta_k: None,
            outdegree_coding: EncodingType::GAMMA, 
            block_coding: EncodingType::GAMMA, 
            residual_coding: EncodingType::ZETA, 
            reference_coding: EncodingType::UNARY, 
            block_count_coding: EncodingType::GAMMA, 
            offset_coding: EncodingType::GAMMA,
            _phantom_block_coding: PhantomData,
            _phantom_block_count_coding: PhantomData,
            _phantom_outdegree_coding: PhantomData,
            _phantom_offset_coding: PhantomData,
            _phantom_reference_coding: PhantomData,
            _phantom_residual_coding: PhantomData,
        }
    }
}

impl<
    BlockCoding: UniversalCode,
    BlockCountCoding: UniversalCode,
    OutdegreeCoding: UniversalCode,
    OffsetCoding: UniversalCode,
    ReferenceCoding: UniversalCode,
    ResidualCoding: UniversalCode,
    T
> BVGraphBuilder<
    BlockCoding,
    BlockCountCoding,
    OutdegreeCoding,
    OffsetCoding,
    ReferenceCoding,
    ResidualCoding,
    T
> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
        + FromStr
        + Display
        + Copy
{
    pub fn new() -> BVGraphBuilder<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding, T> {
        Self::default()
    }

    /// Loads a previously-compressed BVGraph.
    /// 
    /// This method can be called either before or after [`Self::load_offsets()`].
    ///  
    /// # Arguments
    /// 
    /// * `filename` - The base name of the compressed graph file
    /// 
    /// # Examples
    /// ```
    /// let file_base_name = "graph1";
    /// let builder = BVGraphBuilder::<GammaCode, GammaCode, UnaryCode, DeltaCode, GammaCode, GammaCode, usize>::new()
    ///                 .load_properties(file_base_name);
    ///                 .load_graph(file_base_name);
    /// ```
    pub fn load_graph(mut self, filename: &str) -> Self {
        // let deserialized_graph = bincode::deserialize::<Box<[u8]>>(
        //                     fs::read(format!("{}.graph", filename)).unwrap().as_slice()
        //                     ).unwrap().to_vec();
        let deserialized_graph = fs::read(format!("{}.graph", filename)).unwrap();

        self.loaded_graph = deserialized_graph.iter().map(|x| T::from(*x as usize)).collect();

        self
    }

    /// Loads a previously-compressed BVGraph's offsets file.
    /// 
    /// This method can be called either before or after [`Self::load_graph()`].
    ///  
    /// # Arguments
    /// 
    /// * `filename` - The base name of the compressed graph file
    /// 
    /// # Examples
    /// ```
    /// let file_base_name = "graph1";
    /// let builder = BVGraphBuilder::<GammaCode, GammaCode, UnaryCode, DeltaCode, GammaCode, GammaCode, usize>::new()
    ///                 .load_properties(file_base_name);
    ///                 .load_graph(file_base_name);
    ///                 .load_offsets(file_base_name);
    /// let graph = builder.build();
    /// ```
    pub fn load_offsets(mut self, filename: &str) -> Self {
        // let deserialized_offsets = bincode::deserialize::<Box<[u8]>>(
        //                     fs::read(format!("{}.offsets", filename)).unwrap().as_slice()
        //                     ).unwrap().to_vec();
        let deserialized_offsets = fs::read(format!("{}.offsets", filename)).unwrap();

        self.loaded_offsets = deserialized_offsets.iter().map(|x| *x as usize).collect();

        self
    }

    /// Loads a graph file represented in plain mode.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The basename of the compressed graph file
    pub fn load_graph_plain(mut self, filename: &str) -> Self {
        self.loaded_graph = fs::read_to_string(format!("{}.graph.plain", filename))
                            .expect("Failed to load the graph file")
                            .split(' ')
                            .map(|node| node
                                                .parse()
                                                // This should account also for overflows
                                                .unwrap_or_else(|_| panic!("Failed to parse node {}", node))
                            )
                            .collect();

        self
    }

    /// Loads the offsets file represented in plain mode.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The basename of the compressed offsets file
    pub fn load_offsets_plain(mut self, filename: &str) -> Self {

        self.loaded_offsets = fs::read_to_string(format!("{}.offsets.plain", filename))
                            .expect("Failed to load the offsets file")
                            .split(' ')
                            .map(|node| node
                                                .parse()
                                                .unwrap_or_else(|_| panic!("Failed to parse offset {}", node))
                            )
                            .collect();

        // TODO: move into 'count_nodes_plain()' and 'count_edges_plain()' 
        // Since the files are not encoded, this is fine
        self.num_nodes = self.loaded_offsets.len();
        self.num_edges = self.loaded_graph.len() - self.num_nodes * 2;

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
        BlockCoding,
        BlockCountCoding,
        OutdegreeCoding,
        OffsetCoding,
        ReferenceCoding,
        ResidualCoding,
        T
    > {
        BVGraph::<BlockCoding, BlockCountCoding, OutdegreeCoding, OffsetCoding, ReferenceCoding, ResidualCoding, T> { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph.into_boxed_slice(), 
            offsets: self.loaded_offsets, 
            cached_node: self.cached_node, 
            cached_outdegree: self.cached_outdegree, 
            cached_ptr: self.cached_ptr, 
            max_ref_count: self.max_ref_count, 
            window_size: self.window_size,
            min_interval_len: self.min_interval_len,
            zeta_k: self.zeta_k,
            _phantom_block_coding: PhantomData,
            _phantom_block_count_coding: PhantomData,
            _phantom_outdegree_coding: PhantomData,
            _phantom_offset_coding: PhantomData,
            _phantom_reference_coding: PhantomData,
            _phantom_residual_coding: PhantomData
        }
    }
}

#[cfg(test)]
mod tests;