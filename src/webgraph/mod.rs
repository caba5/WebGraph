mod tests;

use std::{fs::{self, File}, io::{Write}, vec, cmp::Ordering};

use serde::{Serialize, Deserialize};

use crate::{ImmutableGraph, EncodingType, uncompressed_graph::UncompressedGraph};

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
        let mut bit_offset = 0;

        let mut bit_count = Vec::<usize>::new();

        let mut graph_file = File::create(format!("{}.graph", filename))?;
        let mut offsets_file = File::create(format!("{}.offsets", filename))?;

        let mut graph_buf = Vec::new();
        let mut offsets_buf = Vec::new();

        let cyclic_buff_size = self.window_size + 1;
        let mut list = vec![vec![0; 1024]; cyclic_buff_size];
        let mut list_len = vec![0; cyclic_buff_size];
        let mut ref_count: Vec<i32> = vec![0; cyclic_buff_size];

        let mut node_iter = self.iter(); /////////////////

        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.next().unwrap();
            let curr_idx = curr_node % cyclic_buff_size;

            self.write_offset(&mut offsets_buf, graph_buf.len() - bit_offset).unwrap();

            bit_offset += graph_buf.len();
            
            self.write_outdegree(&mut graph_buf, outd).unwrap();

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
                let mut best_ref: i32 = -1;
                let mut cand;

                ref_count[curr_idx] = -1;

                for r in 0..cyclic_buff_size {
                    cand = ((curr_node - r + cyclic_buff_size) % cyclic_buff_size) as i32;
                    if ref_count[cand as usize] < (self.max_ref_count as i32) && list_len[cand as usize] != 0 {
                        let diff_comp = 
                            self.diff_comp(&mut bit_count, 
                                            curr_node, 
                                            r, 
                                            list[cand as usize].clone(), 
                                            list_len[cand as usize], 
                                            list[curr_idx].clone(), 
                                            list_len[curr_idx]).unwrap(); // TODO: manage?
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
                    &mut graph_buf, 
                    curr_node, 
                    best_ref as usize, 
                    list[best_cand as usize].clone(), 
                    list_len[best_cand as usize], 
                    list[curr_idx].clone(), 
                    list_len[curr_idx]
                ).unwrap(); // TODO: manage?
            }
        }
        
        // We write the final offset to the offset stream
        self.write_offset(&mut offsets_buf, graph_buf.len() - bit_offset).unwrap(); // TODO: manage?

        // Temporary
        let graph_buf = unsafe {
            std::slice::from_raw_parts(
                graph_buf.as_ptr() as *const u8, 
                graph_buf.len() * std::mem::size_of::<usize>()
            )
        };

        graph_file.write_all(graph_buf)?; // TODO: manage?

        let offsets_buf = unsafe {
            std::slice::from_raw_parts(
                offsets_buf.as_ptr() as *const u8, 
                offsets_buf.len() * std::mem::size_of::<usize>()
            )
        };

        offsets_file.write_all(offsets_buf)?; // TODO: manage?
        Ok(())
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
#[derive(Debug)]
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
        if x > self.n - 1 {
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

    fn write_reference(&self, reference_out: &mut Vec<usize>, reference: usize) -> Result<usize, &str> {
        if reference > self.window_size {
            return Err("The required reference is incompatible with the window size");
        }
        reference_out.push(reference);
        Ok(reference)
        // TODO: implement coded writing
    }

    fn write_outdegree(&self, outdegree_out: &mut Vec<usize>, outdegree: usize) -> Result<usize, &str> {
        outdegree_out.push(outdegree);
        Ok(outdegree)
        // TODO: implement coded writing
    }

    fn write_block_count(&self, block_count_out: &mut Vec<usize>, block_count: usize) -> Result<usize, &str> {
        block_count_out.push(block_count);
        Ok(block_count)
        // TODO: implement coded writing
    }

    fn write_block(&self, block_out: &mut Vec<usize>, block: usize) -> Result<usize, &str> {
        block_out.push(block);
        Ok(block)
        // TODO: implement coded writing
    }

    fn write_residual(&self, residual_out: &mut Vec<usize>, residual: usize) -> Result<usize, &str> {
        residual_out.push(residual);
        Ok(residual)
        // TODO: implement coded writing
    }

    fn write_offset(&self, offset_out: &mut Vec<usize>, offset: usize) -> Result<usize, &str> {
        offset_out.push(offset);
        Ok(offset)
        // TODO: implement coded writing
    }

    fn intervallize(
        &self, 
        x: &Vec<usize>,
        left: &mut Vec<usize>, 
        len: &mut Vec<usize>, 
        residuals: &mut Vec<usize>
    ) -> usize {
        let mut n_interval = 0;
        let v1 = x.len();
        let v = x.clone();

        let mut j;

        left.clear();
        len.clear();
        residuals.clear();

        for mut i in 0..v1 {
            j = 0;
            if i < v1 - 1 && v[i] + 1 == v[i + 1] {
                j += 1;
                while i + j < v1 - 1 && v[i + j] + 1 == v[i + j + 1 ] {
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
        }

        n_interval
    }
    
    fn diff_comp(
        &self,
        output_stream: &mut Vec<usize>,
        curr_node: usize, 
        reference: usize, 
        ref_list: Vec<usize>, 
        mut ref_len: usize, 
        curr_list: Vec<usize>, 
        curr_len: usize
    ) -> Result<usize, String> {
        let mut blocks = Vec::<usize>::new();
        let mut extras = Vec::<usize>::new();
        let mut left = Vec::<usize>::new();
        let mut len = Vec::<usize>::new();
        let mut residuals = Vec::<usize>::new();

        let written_data_at_start = output_stream.len();

        let mut i: usize;
        let mut t;
        let mut j = 0; // index of the next successor of the current node we must examine
        let mut k = 0; // index of the next successor of the reference node we must examine
        let mut prev = 0;
        let mut curr_block_len = 0; //number of entries (in the reference list) we have already copied/ignored (in the current block)

        let mut copying = true; // true iff we are producing a copy block (instead of an ignore block)

        if reference == 0 {
            ref_len = 0;
        }

        blocks.clear();
        extras.clear();

        while j < curr_len && k < ref_len {
            if copying { // First case: we are currently copying entries from the reference list
                match curr_list[j].cmp(&curr_list[k]) {
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
        and did not copy up to the end of th eref list */
        if copying && k < ref_len {
            blocks.push(curr_block_len);
        }

        // If there are still missing elements add them to the extra list
        while j < curr_len {
            extras.push(curr_list[j]);
            j += 1;
        }

        let block = blocks.clone(); // ?
        let block_count = blocks.len();
        let extra_count = extras.len();

        // If we have a nontrivial reference window we write the reference to the reference list
        if self.window_size > 0 {
            t = self.write_reference(output_stream, reference)?;
        }

        // Then, if the reference is not void we write the length of the copy list
        if reference != 0 {
            t = self.write_block_count(output_stream, block_count)?;

            // Then, we write the copy list; all lengths except the first one are decremented
            if block_count > 0 {
                t = self.write_block(output_stream, block[0])?;
                for i in 1..block_count {
                    t = self.write_block(output_stream, block[i])?;
                }
            }
        }

        // Finally, we write the extra list
        if extra_count > 0 {
            let residual;
            let residual_count;

            if self.min_interval_len != 0 {
                // If we are to produce intervals, we first compute them
                let interval_count = self.intervallize(&extras, &mut left, &mut len, &mut residuals);

                // Should've been a writeGamma !!!
                output_stream.push(interval_count);
                t = interval_count;

                let mut curr_int_len;

                for i in 0..interval_count {
                    if i == 0 {
                        // Should've been a "writeLongGamma" !!!
                        prev = left[i];
                        output_stream.push(prev - curr_node);
                        t = prev - curr_node;
                    } else {
                        // Should've been a writeGamma !!!
                        output_stream.push(left[i] - prev - i);
                        t = left[i] - prev - i;
                    }
                    
                    curr_int_len = len[i];

                    prev = left[i] + curr_int_len;

                    // Should've been a writeGamma !!!
                    output_stream.push(curr_int_len - self.min_interval_len);
                    t = curr_int_len - self.min_interval_len;
                }

                residual = residuals.clone();
                residual_count = residuals.len();
            } else {
                residual = extras.clone();
                residual_count = extras.len();
            }

            // Now we write out the residuals, if any
            if residual_count != 0 {
                prev = residual[0];
                t = self.write_residual(output_stream, prev - curr_node)?;
                for i in 1..residual_count {
                    if residual[i] == prev {
                        return Err(format!("Repeated successor {} in successor list of node {}", prev, curr_node));
                    }
                    t = self.write_residual(output_stream, residual[i] - prev - 1)?;
                    prev = residual[i];
                }
            }
        }

        Ok(output_stream.len() - written_data_at_start)
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

impl<T> From<UncompressedGraph<T>> for BVGraphBuilder
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
{
    fn from(graph: UncompressedGraph<T>) -> Self {
        let mut graph_with_outdegrees = Vec::new();
        graph_with_outdegrees.reserve(graph.graph_memory.len());

        let mut n = 0;

        for (i, x) in graph.graph_memory.iter().enumerate() {
            graph_with_outdegrees.push(x.to_usize().unwrap());
            if n == 0 || graph.offsets[n - 1] == i {
                let outd = graph.outdegree_internal(n.into());
                graph_with_outdegrees.push(outd);
                n += 1;
            }
        }

        let mut new_offsets = Vec::new();
        new_offsets.reserve(graph.offsets.len());

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