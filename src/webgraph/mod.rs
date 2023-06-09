mod tests;

use std::{fs::{self, File}, io::{BufReader, Read}};

use serde::{Serialize, Deserialize};

use crate::{ImmutableGraph, EncodingType, Properties, uncompressed_graph::UncompressedGraph};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraph {
    n: usize,
    m: usize,
    graph_memory: Vec<usize>,  // TODO: is it on T?
    offsets: Vec<usize>,  // TODO: it is converted from an EliasFanoLongMonotoneList
    cached_node: Option<usize>,
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

impl ImmutableGraph for BVGraph {
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

        let mut node_iter = self.iter();
        node_iter.position_to(self.offsets[x]).ok()?;

        self.cached_outdegree = self.read_outdegree(&mut node_iter);

        self.cached_outdegree
    }

    fn store(&self, filename: &str) -> std::io::Result<()> {
        todo!()
    }
}

/// Defines an iterator over all the elements of the graph vector.
pub struct BVGraphIterator<BV: AsRef<BVGraph>> {
    curr: usize,
    graph: BV
}

impl<BV: AsRef<BVGraph>> Iterator for BVGraphIterator<BV> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.graph.as_ref().graph_memory.len() {
            return None;
        }

        self.curr += 1;

        Some(self.graph.as_ref().graph_memory[self.curr - 1])
    }
}

impl<BV: AsRef<BVGraph>> BVGraphIterator<BV> {
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
    /// graph.iter().position_to(7);  // Where graph is a BVGraph with >7 elements in its graph vector
    fn position_to(&mut self, n: usize) -> Result<(), &str> {
        if n > self.graph.as_ref().graph_memory.len() - 1 {
            return Err("The provided position exceeds the number of possible positions in the graph vector");
        }

        if n < self.curr {
            return Err("The provided position comes before the current position of the iterator");
        }

        while self.curr < n {
            self.curr += 1;
        }

        Ok(())

    }
}

/// Defines an iterator over the successors of a node in a graph.
pub struct BVGraphSuccessorsIterator<BV: AsRef<BVGraph>> {
    base: usize,
    idx_from_base: usize,
    up_to: usize,
    graph: BV
}

impl<BV: AsRef<BVGraph>> Iterator for BVGraphSuccessorsIterator<BV> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base + self.idx_from_base > self.up_to {
            return None;
        }

        let g = self.graph.as_ref();
        
        self.idx_from_base += 1;        
        
        Some(g.graph_memory[self.base + self.idx_from_base].clone())
    }
}

impl AsRef<BVGraph> for BVGraph {
    fn as_ref(&self) -> &BVGraph {
        self
    }
}

impl IntoIterator for BVGraph {
    type Item = usize;

    type IntoIter = BVGraphIterator<BVGraph>;

    fn into_iter(self) -> Self::IntoIter {
        BVGraphIterator {
            curr: 0,
            graph: self
        }
    }
}

impl BVGraph {
    pub fn iter(&self) -> BVGraphIterator<&BVGraph> {
        BVGraphIterator {
            curr: 0,
            graph: self
        }
    }

    pub fn successors_iter(&self) -> BVGraphSuccessorsIterator<&BVGraph> {
        BVGraphSuccessorsIterator { 
            base: 0, 
            idx_from_base: 0, 
            up_to: self.graph_memory.len(), 
            graph: self 
        }
    }

    fn successors(&self, x: u8) -> Result<Box<dyn Iterator<Item = &u32>>, &str> {
        if x as usize > self.n {
            return  Err("Node index out of range");
        }

        let graph_iter = self.graph_memory.iter();  // TODO: need to iterate on bits
        todo!()
        // return self.successors_internal(x, &graph_iter, Option::None, Option::None);
    }

    fn read_outdegree(&self, outdegree_iter: &mut BVGraphIterator<&Self>) -> Option<usize> { 
        outdegree_iter.next()
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

    pub fn load(name: &str) -> BVGraph {
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

pub struct BVGraphBuilder {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<usize>,   // TODO: does this BVGraph implementation have to deal with generics instead of bytes?
    loaded_offsets: Vec<usize>,
    cached_node: Option<usize>,
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

// TODO: this method is wrong since BVGraph expects to have each node followed by its
// outdegree, while the uncompressed graph calculates the outdegree for each node.
// impl<T> From<UncompressedGraph<T>> for BVGraphBuilder
// where T:
//         num_traits::Num 
//         + PartialOrd 
//         + num_traits::ToPrimitive
//         + serde::Serialize
// {
//     fn from(graph: UncompressedGraph<T>) -> Self {
//         Self { 
//             num_nodes: graph.num_nodes(), 
//             num_edges: graph.num_arcs(), 
//             loaded_graph: graph.graph_memory.iter()
//                                             .map(|val| val.to_usize().unwrap())
//                                             .collect(),
//             loaded_offsets: graph.offsets, 
//             cached_node: None, 
//             cached_outdegree: None, 
//             cached_ptr: None, 
//             max_ref_count: 0, 
//             window_size: 0, 
//             min_interval_len: 0, 
//             zeta_k: 0, 
//             outdegree_coding: EncodingType::GAMMA, 
//             block_coding: EncodingType::GAMMA, 
//             residual_coding: EncodingType::ZETA, 
//             reference_coding: EncodingType::UNARY, 
//             block_count_coding: EncodingType::GAMMA, 
//             offset_coding: EncodingType::GAMMA 
//         }
//     }
// }

impl Default for BVGraphBuilder {
    fn default() -> Self {
        BVGraphBuilder::new()
    }
}

impl BVGraphBuilder {
    pub fn new() -> BVGraphBuilder {
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

        self.loaded_graph = buf.into_iter().map(usize::from).collect();

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

        // TODO: Remove
        // For now, since the files are not encoded, this should be fine
        self.num_nodes = self.loaded_offsets.len();
        self.num_edges = self.loaded_graph.len() - self.num_nodes * 2;

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

    /// Constructs the BVGraph object.
    pub fn construct(self) -> BVGraph {
        BVGraph { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph, 
            offsets: self.loaded_offsets, 
            cached_node: self.cached_node, 
            cached_outdegree: self.cached_outdegree, 
            cached_ptr: self.cached_ptr, 
            max_ref_count: self.max_ref_count, 
            window_size: self.window_size, 
            min_interval_len: self.min_interval_len, 
            zeta_k: self.zeta_k, 
            outdegree_coding: self.outdegree_coding, 
            block_coding: self.block_coding, 
            residual_coding: self.residual_coding, 
            reference_coding: self.reference_coding, 
            block_count_coding: self.block_count_coding, 
            offset_coding: self.offset_coding 
        }
    }
}