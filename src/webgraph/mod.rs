mod tests;

use std::{fs::{self, File}, io::{BufReader, Read, Write}, vec};

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
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1]}).ok()?;

        self.cached_outdegree = self.read_outdegree(&mut node_iter);

        self.cached_outdegree
    }

    fn store(&self, filename: &str) -> std::io::Result<()> {
        let outd: usize;
        let curr_idx: usize;

        let mut graph_file = File::create(format!("{}.graph", filename))?;
        let mut offsets_file = File::create(format!("{}.offsets", filename))?;

        let mut graph_buf = Vec::new();
        let mut offsets_buf = Vec::new();

        let cyclic_buff_size = self.window_size + 1;
        let mut list = vec![vec![0; 1024]; cyclic_buff_size];
        let mut list_len = vec![0; cyclic_buff_size];
        let mut ref_count: Vec<i32> = vec![0; cyclic_buff_size];

        let updates = 0;

        let mut node_iter = self.iter();

        let mut written = 0;

        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.next().unwrap();
            let curr_idx = curr_node % cyclic_buff_size;

            offsets_buf.push(written);
            
            graph_buf.push(outd); //

            if outd > list[curr_idx].len() {
                list[curr_idx].resize(outd, 0);
            }

            let mut successors = Vec::new();
            let mut successors_it = self.successors(curr_node).unwrap();

            while successors_it.has_next() {
                successors.push(successors_it.next().unwrap());
            }
            
            list_len[curr_idx] = outd;
            
            if outd > 0 {
                let mut best_comp = i64::MAX;
                let mut best_cand = -1;
                let mut best_ref = -1;
                let mut cand = -1;

                ref_count[curr_idx] = -1;

                for r in 0..cyclic_buff_size {
                    cand = ((curr_node - r + cyclic_buff_size) % cyclic_buff_size) as i32;
                    if ref_count[cand as usize] < (self.max_ref_count as i32) && list_len[cand as usize] != 0 {
                        let diff_comp = todo!();
                    }
                }

                
            }

        }

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
        if self.curr > self.graph.as_ref().graph_memory.len() {
            return None;
        }

        self.curr += 1;

        Some(self.graph.as_ref().graph_memory[self.curr])
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

        while self.curr < n {
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
pub struct BVGraphSuccessorsIterator<BV: AsRef<BVGraph>> {
    base: usize,
    idx_from_base: usize,
    up_to: usize,
    graph: BV
}

impl<BV: AsRef<BVGraph>> Iterator for BVGraphSuccessorsIterator<BV> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.has_next() {
            return None;
        }

        let g = self.graph.as_ref();
        
        self.idx_from_base += 1;        
        
        Some(g.graph_memory[self.base + self.idx_from_base])
    }
}

impl<BV: AsRef<BVGraph>> BVGraphSuccessorsIterator<BV> {
    /// Returns `true` if the iterator has not reached the final successor of the node, `false` otherwise.
    #[inline]
    fn has_next(&self) -> bool {
        self.base + self.idx_from_base + 1 < self.up_to
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
    
    fn outdegree_internal(&self, x: usize) -> usize {
        let mut node_iter = self.iter();
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1]}).ok();
        self.read_outdegree(&mut node_iter).unwrap()
    }

    // TODO
    fn successors(&self, x: usize) -> Option<BVGraphSuccessorsIterator<&BVGraph>> {
        if x > self.n {
            return None;
        }

        let base = if x == 0 {0} else { self.offsets[x - 1] };
        Some(BVGraphSuccessorsIterator {
            base,
            idx_from_base: 1, // starts from the outdeg
            up_to: base + self.outdegree_internal(x) + 2, // summing 2 to skip the node and its outdeg
            graph: self
        })
    }

    fn read_outdegree(&self, outdegree_iter: &mut BVGraphIterator<&Self>) -> Option<usize> {
        outdegree_iter.next()
        // TODO: implement outdegree_iter.read_gamma()
        // TODO: implement outdegree_iter.read_delta()
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

    /// Loads a graph file represented in plain mode.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the compressed graph file
    pub fn load_graph(mut self, filename: &str) -> Self {
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
    /// * `filename` - The filename of the compressed offsets file
    pub fn load_offsets(mut self, filename: &str) -> Self {

        self.loaded_offsets = fs::read_to_string(format!("{}.offsets.plain", filename))
                            .expect("Failed to load the offsets file")
                            .split(' ')
                            .map(|node| node
                                                .parse()
                                                .unwrap_or_else(|_| panic!("Failed to parse offset {}", node))
                            )
                            .collect();

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