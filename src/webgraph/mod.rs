mod tests;

use std::{fs::{self, File}, io::{BufReader, Read}, marker::PhantomData};

use serde::{Serialize, Deserialize};

use crate::{ImmutableGraph, EncodingType, Properties, uncompressed_graph::UncompressedGraph};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraph<T> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize 
{
    n: usize,
    m: usize,
    graph_memory: Vec<u8>,  // TODO: is it on T?
    offsets: Vec<u64>,  // TODO: it is converted from an EliasFanoLongMonotoneList
    cached_node: T,
    cached_outdegree: usize,
    cached_ptr: usize,
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: usize,
    outdegree_coding: EncodingType,
    block_coding: EncodingType,
    residual_coding: EncodingType,
    reference_coding: EncodingType,
    block_count_coding: EncodingType,
    offset_coding: EncodingType,
}

impl<T> ImmutableGraph for BVGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize 
{
    type NodeT = T;

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

    /// Returns the outdegree of a given node.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize> {
        if x == self.cached_node {
            return Some(self.cached_outdegree);
        }
        
        let usized_x = x.to_usize().unwrap();
        if x < T::zero() || usized_x >= self.n {
            return None;
        }

        self.cached_node = x;

        let node_iter = self.graph_memory.iter().nth(self.offsets[usized_x] as usize).unwrap();

        self.cached_outdegree = self.read_outdegree(node_iter);

        Some(self.cached_outdegree)
    }

    fn store(&self, filename: &str) -> std::io::Result<()> {
        todo!()
    }
}

pub struct BVGraphIterator<T, BV: AsRef<BVGraph<T>>> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize 
{
    base: usize,
    idx_from_base: usize,
    up_to: usize,
    graph: BV,
    _phantom: PhantomData<T>
}

impl<T, BV: AsRef<BVGraph<T>>> Iterator for BVGraphIterator<T, BV>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive 
        + serde::Serialize
        + Clone
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base + self.idx_from_base > self.up_to {
            return None;
        }

        let g = self.graph.as_ref();
        
        self.idx_from_base += 1;        
        
        Some(g.graph_memory[self.base + self.idx_from_base].clone())
    }
}

impl<T> AsRef<BVGraph<T>> for BVGraph<T>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    fn as_ref(&self) -> &BVGraph<T> {
        self
    }
}

impl<T> IntoIterator for BVGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
{
    type Item = T;

    type IntoIter = BVGraphIterator<T, BVGraph<T>>;

    fn into_iter(self) -> Self::IntoIter {
        BVGraphIterator {
            base: 0,
            idx_from_base: 0,
            up_to: self.graph_memory.len(),
            graph: self,
            _phantom: PhantomData
        }
    }
}

impl<T> BVGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    fn successors(&self, x: T) -> Result<Box<dyn Iterator<Item = &u32>>, &str> {
        if x < T::zero() || x.to_usize().unwrap() > self.n {
            return  Err("Node index out of range");
        }

        let graph_iter = self.graph_memory.iter();  // TODO: need to iterate on bits
        todo!()
        // return self.successors_internal(x, &graph_iter, Option::None, Option::None);
    }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_outdegree<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> u32 { 
        
        // TODO: implement outdegree_iter.read_gamma()
        // TODO: implement outdegree_iter.read_delta()
    }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_block<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> Result<u32, &str> { // TODO: better error
        match self.outdegree_coding {
            EncodingType::UNARY => todo!(), // TODO: implement outdegree_iter.read_unary()
            EncodingType::GAMMA => todo!(),  // TODO: implement outdegree_iter.read_gamma()
            EncodingType::DELTA => todo!(),   // TODO: implement outdegree_iter.read_delta()
            _ => Err("The encoding is not supported")
        }
    }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_block_count<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> Result<u32, &str> { // TODO: better error
        match self.outdegree_coding {
            EncodingType::UNARY => todo!(), // TODO: implement outdegree_iter.read_unary()
            EncodingType::GAMMA => todo!(),  // TODO: implement outdegree_iter.read_gamma()
            EncodingType::DELTA => todo!(),   // TODO: implement outdegree_iter.read_delta()
            _ => Err("The encoding is not supported")
        }
    }

    // TODO: rename outdegree_iter to input_bit_stream?
    fn read_reference<'a>(&self, outdegree_iter: &impl Iterator<Item = &'a u8>) -> Result<u32, &str> { // TODO: better error
        let refer: u32 = match self.reference_coding {
            EncodingType::UNARY => todo!(), // TODO: implement outdegree_iter.read_unary()
            EncodingType::GAMMA => todo!(),  // TODO: implement outdegree_iter.read_gamma()
            EncodingType::DELTA => todo!(),   // TODO: implement outdegree_iter.read_delta()
            _ => return Err("The encoding is not supported")
        };

        // if refer > self.window_size {
        //     return Err("The required reference is incompatible with the windows size");
        // }

        Ok(refer)
    }

    pub fn load(name: &str) -> BVGraph<T> {
        // Read properties file. TODO: how to represent it (JSON)?
        let graph_props = fs::read(format!("{}.properties", &name)).unwrap();
        let graph_props: Properties = serde_json::from_slice(&graph_props.as_slice()).unwrap();
        
        // Create an input stream to the graph file
        let graph_f = File::open(format!("{}.graph", &name)).unwrap();

        let file_size = graph_f.metadata().unwrap().len();

        let mut graph_memory: Vec<u8>;
        if file_size <= todo!() { 
            graph_memory = Vec::with_capacity(usize::try_from(file_size).unwrap());
            let mut file_reader = BufReader::new(graph_f);
            file_reader.read_to_end(&mut graph_memory).unwrap(); // Should read the whole graph into memory. Potentially inefficient
        } else {
            todo!() // WebGraph uses "FastMultiByteArrayInputStream" that multiplexes the input content across multiple arrays
        }

        // Create an input stream to the offsets
        let offsets_f = File::open(format!("{}.offsets", &name)).unwrap();
        let offsets_reader = BufReader::new(offsets_f);
        // TODO: It wraps the input stream into an 'EliasFanoMonotoneLongBigList" https://github.com/vigna/Sux4J/blob/master/src/it/unimi/dsi/sux4j/util/EliasFanoMonotoneLongBigList.java#L313

        // TODO: Need to create an 'InputBitStream' around graph_memory

        todo!()
    }


    pub fn store(&self, name: &str) {

        // ...compression perfomed by threads...

        // Writes graph properties in JSON (JSON?)
        let props = Properties{ 
            nodes: self.n, 
            arcs: self.m, 
            window_size: self.window_size, 
            max_ref_count: self.max_ref_count, 
            min_interval_len: self.min_interval_len, 
            zetak: self.zeta_k, 
            tot_links: 0,
            comp: EncodingType::ZETA, 
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

        let json_props = serde_json::to_string(&props).unwrap(); // TODO: should it return a Result?
        fs::write(format!("{}.properties", &name), json_props).unwrap();
        // fs::write(format!("{}.graph", &name), bincode::serialize(self).unwrap()).unwrap(); // TODO
    }
    
}

pub struct BVGraphBuilder<T> {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<u8>,   // TODO: does this BVGraph implementation have to deal with generics instead of bytes?
    loaded_offsets: Vec<usize>,
    cached_node: Option<T>,
    cached_outdegree: Option<usize>,
    cached_ptr: Option<usize>,
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: usize,
    outdegree_coding: EncodingType,
    block_coding: EncodingType,
    residual_coding: EncodingType,
    reference_coding: EncodingType,
    block_count_coding: EncodingType,
    offset_coding: EncodingType,
}

impl<T> From<UncompressedGraph<T>> for BVGraphBuilder<T> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    fn from(graph: UncompressedGraph<T>) -> Self {
        Self { 
            num_nodes: graph.num_nodes(), 
            num_edges: graph.num_arcs(), 
            loaded_graph: graph.graph_memory.iter()
                                            .map(|val| val.to_u8().unwrap())
                                            .collect(),
            loaded_offsets: graph.offsets, 
            cached_node: None, 
            cached_outdegree: None, 
            cached_ptr: None, 
            max_ref_count: 0, 
            window_size: 0, 
            min_interval_len: 0, 
            zeta_k: 0, 
            outdegree_coding: EncodingType::GAMMA, 
            block_coding: EncodingType::GAMMA, 
            residual_coding: EncodingType::ZETA, 
            reference_coding: EncodingType::UNARY, 
            block_count_coding: EncodingType::GAMMA, 
            offset_coding: EncodingType::GAMMA 
        }
    }
}

impl<T> Default for BVGraphBuilder<T> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    fn default() -> Self {
        BVGraphBuilder::<T>::new()
    }
}

impl<T> BVGraphBuilder<T> 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    pub fn new() -> BVGraphBuilder<T> {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Vec::new(), 
            loaded_offsets: Vec::new(), 
            cached_node: None, 
            cached_outdegree: None, 
            cached_ptr: None, 
            max_ref_count: 0, 
            window_size: 0, 
            min_interval_len: 0, 
            zeta_k: 0, 
            outdegree_coding: EncodingType::GAMMA, 
            block_coding: EncodingType::GAMMA, 
            residual_coding: EncodingType::ZETA, 
            reference_coding: EncodingType::UNARY, 
            block_count_coding: EncodingType::GAMMA, 
            offset_coding: EncodingType::GAMMA
        }
    }

    /// Loads a graph file represented in binary.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the compressed graph file
    pub fn load_graph(mut self, filename: &str) -> Self {
        let f = File::open(filename).expect("Failed to open the graph file");
        let mut reader = BufReader::new(&f);

        let mut buf = Vec::<u8>::new();
        buf.reserve(f.metadata().unwrap().len() as usize);

        reader.read_to_end(&mut buf).expect("Failed in reading the graph file");

        self.loaded_graph = buf;

        self
    }

    /// Loads the offsets file represented in binary.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the compressed offsets file
    pub fn load_offsets(mut self, filename: &str) -> Self {
        let f = File::open(filename).expect("Failed to open the offsets file");
        let mut reader = BufReader::new(&f);

        let mut buf = Vec::<u8>::new();
        buf.reserve(f.metadata().unwrap().len() as usize);

        reader.read_to_end(&mut buf).expect("Failed in reading the offsets file");

        self.loaded_offsets = buf.into_iter().map(usize::from).collect();

        self
    }

    // TODO: Is this necessary?
    pub fn load_properties(mut self, filename: &str) -> Self {
        todo!()
    }

    /// Sets the maximum reference chain length.
    /// 
    /// # Arguments
    /// 
    /// * `ref_count` - The maximum length of the chain.
    pub fn set_ref_count(mut self, ref_count: usize) -> Self {
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

    /// Sets the *k* parameter for *zeta*-coding.
    /// 
    /// # Arguments
    /// 
    /// * `zeta_k` - The *k* parameter.
    pub fn set_zeta_k(mut self, zeta_k: usize) -> Self {
        self.zeta_k = zeta_k;

        self
    }

    /// Sets the encoding type for outdegrees.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_outdegree_coding(mut self, code: EncodingType) -> Self {
        self.outdegree_coding = code;

        self
    }
    
    /// Sets the encoding type for blocks.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_block_coding(mut self, code: EncodingType) -> Self {
        self.block_coding = code;

        self
    }
    
    /// Sets the encoding type for the lists of residuals.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_residual_coding(mut self, code: EncodingType) -> Self {
        self.residual_coding = code;

        self
    }

    /// Sets the encoding type for references.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_reference_coding(mut self, code: EncodingType) -> Self {
        self.reference_coding = code;

        self
    }

    /// Sets the encoding type for the block counters.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_block_count_coding(mut self, code: EncodingType) -> Self {
        self.block_count_coding = code;

        self
    }

    /// Sets the encoding type for the offsets.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_offset_coding(mut self, code: EncodingType) -> Self {
        self.offset_coding = code;

        self
    }
}