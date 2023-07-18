mod tests;

use std::{fs, vec, cmp::Ordering};

use num_traits::ToPrimitive;
use serde::{Serialize, Deserialize};
use sucds::{mii_sequences::{EliasFanoBuilder, EliasFano}, Serializable};

use crate::{ImmutableGraph, Properties, EncodingType};
use crate::uncompressed_graph::UncompressedGraph;
use crate::bitstreams::{InputBitStream, OutputBitStreamBuilder};

#[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
struct BVGraphParameters {
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

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraph {
    n: usize,
    m: usize,
    graph_memory: Box<[u8]>,  // TODO: is it on T?
    offsets: Box<[u8]>,  // TODO: it is converted from an EliasFanoLongMonotoneList
    cached_node: Option<usize>,
    cached_outdegree: Option<usize>,
    cached_ptr: Option<usize>,
    parameters: BVGraphParameters
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
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1] as usize}).ok()?;

        self.cached_outdegree = self.read_outdegree(&mut node_iter);

        self.cached_outdegree
    }

    fn store(&self, filename: &str) -> std::io::Result<()> {
        let mut bit_offset = 0;

        let mut bit_count = OutputBitStreamBuilder::new();

        // let mut graph_buf = Vec::default();
        // let mut offsets_buf = Vec::default();

        let mut graph_obs = OutputBitStreamBuilder::new();
        let mut offsets_obs = OutputBitStreamBuilder::new();

        let cyclic_buff_size = self.parameters.window_size + 1;
        let mut list = vec![vec![0; 1024]; cyclic_buff_size];
        let mut list_len = vec![0; cyclic_buff_size];
        let mut ref_count: Vec<i32> = vec![0; cyclic_buff_size];

        let mut node_iter = self.iter();
        
        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.next().unwrap();
            let curr_idx = curr_node % cyclic_buff_size;
            
            println!("Curr node: {}", curr_node);
            
            // Doesn't use delta (graph_buf.len() - bit_offset) since it has to be monotonically increasing for EliasFano
            self.write_offset(&mut offsets_obs, graph_obs.written_bits - bit_offset).unwrap();
            
            bit_offset = graph_obs.written_bits;
            
            self.write_outdegree(&mut graph_obs, outd).unwrap();

            if outd > list[curr_idx].len() {
                list[curr_idx].resize(outd, 0);
            }
            
            let mut successors = Vec::default();
            let mut successors_it = self.successors(curr_node).unwrap();

            while successors_it.has_next() {
                successors.push(successors_it.next().unwrap());
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
                    cand = ((curr_node + cyclic_buff_size - r) % cyclic_buff_size) as i32;
                    if ref_count[cand as usize] < (self.parameters.max_ref_count as i32) && list_len[cand as usize] != 0 {
                        let diff_comp = 
                            self.diff_comp(&mut bit_count, 
                                            curr_node, 
                                            r, 
                                            list[cand as usize].as_slice(),
                                            list[curr_idx].as_slice()
                            ).unwrap(); // TODO: manage?
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
                    curr_node, 
                    best_ref as usize, 
                    list[best_cand as usize].as_slice(), 
                    list[curr_idx].as_slice(),
                ).unwrap(); // TODO: manage?
            }
            
            node_iter.advance_by(outd).unwrap();
        }
        
        // We write the final offset to the offset stream
        self.write_offset(&mut offsets_obs, graph_obs.written_bits - bit_offset).unwrap();
        
        // TODO: manage elias-fano offsets writing
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

        let graph = graph_obs.build();
        let offsets = offsets_obs.build();

        fs::write(format!("{}.offsets", filename), bincode::serialize(&offsets.os).unwrap()).unwrap();
        fs::write(format!("{}.graph", filename), bincode::serialize(&graph.os).unwrap()).unwrap();

        let props = Properties {
            nodes: self.n,
            arcs: self.m,
            window_size: self.parameters.window_size,
            max_ref_count: self.parameters.max_ref_count,
            min_interval_len: self.parameters.min_interval_len,
            zeta_k: self.parameters.zeta_k,
            outdegree_coding: self.parameters.outdegree_coding,
            block_coding: self.parameters.block_coding,
            residual_coding: self.parameters.residual_coding,
            reference_coding: self.parameters.reference_coding,
            block_count_coding: self.parameters.block_count_coding,
            offset_coding: self.parameters.offset_coding,
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

        let res = Some(self.graph.as_ref().graph_memory[self.curr] as usize);

        self.curr += 1;

        res
    }
    // fn outdegrees(&self) -> iter;
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
    #[inline(always)]
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
        
        Some(g.graph_memory[self.base + self.idx_from_base] as usize)
    }
}

impl<BV: AsRef<BVGraph>> BVGraphSuccessorsIterator<BV> {
    /// Returns `true` if the iterator has not reached the final successor of the node, `false` otherwise.
    #[inline(always)]
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
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1] as usize}).ok();
        self.read_outdegree(&mut node_iter).unwrap()
    }

    // TODO
    fn successors(&self, x: usize) -> Option<BVGraphSuccessorsIterator<&BVGraph>> {
        if x > self.n - 1 {
            return None;
        }

        let base = if x == 0 {0} else {self.offsets[x - 1]} as usize;
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

    fn write_reference(&self, graph_obs: &mut OutputBitStreamBuilder, reference: usize) -> Result<usize, String> {
        if reference > self.parameters.window_size {
            return Err("The required reference is incompatible with the window size".to_string());
        }
        match self.parameters.reference_coding {
            EncodingType::UNARY => graph_obs.write_unary(reference as u64),
            EncodingType::GAMMA => graph_obs.write_gamma(reference as u64),   // TODO: does the param have to be usize?
            EncodingType::DELTA => graph_obs.write_delta(reference as u64),
            other => return Err(format!("The {} reference coding is not supported", other.clone())),
        };
        Ok(reference)
    }

    fn write_outdegree(&self, graph_obs: &mut OutputBitStreamBuilder, outdegree: usize) -> Result<usize, String> {
        match self.parameters.outdegree_coding {
            EncodingType::GAMMA => graph_obs.write_gamma(outdegree as u64),   // TODO: does the param have to be usize?
            EncodingType::DELTA => graph_obs.write_delta(outdegree as u64),
            other => return Err(format!("The {} outdegree coding is not supported", other)),
        };
        Ok(outdegree)
    }

    fn write_block_count(&self, graph_obs: &mut OutputBitStreamBuilder, block_count: usize) -> Result<usize, String> {
        match self.parameters.block_count_coding {
            EncodingType::UNARY => graph_obs.write_unary(block_count as u64),
            EncodingType::GAMMA => graph_obs.write_gamma(block_count as u64),   // TODO: does the param have to be usize?
            EncodingType::DELTA => graph_obs.write_delta(block_count as u64),
            other => return Err(format!("The {} block coding is not supported", other)),
        };
        Ok(block_count)
    }

    fn write_block(&self, graph_obs: &mut OutputBitStreamBuilder, block: usize) -> Result<usize, String> {
        match self.parameters.block_coding {
            EncodingType::UNARY => graph_obs.write_unary(block as u64),
            EncodingType::GAMMA => graph_obs.write_gamma(block as u64),   // TODO: does the param have to be usize?
            EncodingType::DELTA => graph_obs.write_delta(block as u64),
            other => return Err(format!("The {} block coding is not supported", other)),
        };
        Ok(block)
    }

    fn write_residual(&self, graph_obs: &mut OutputBitStreamBuilder, residual: usize) -> Result<usize, String> {
        match self.parameters.residual_coding {
            EncodingType::GAMMA => graph_obs.write_gamma(residual as u64),   // TODO: does the param have to be usize?
            EncodingType::DELTA => graph_obs.write_delta(residual as u64),
            EncodingType::ZETA => graph_obs.write_zeta(residual as u64, self.parameters.zeta_k as u64), // TODO: zeta_k type?
            other => return Err(format!("The {} residual coding is not supported", other)),
        };
        Ok(residual)
    }

    fn write_offset(&self, offset_obs: &mut OutputBitStreamBuilder, offset: usize) -> Result<usize, String> {
        match self.parameters.offset_coding {
            EncodingType::GAMMA => offset_obs.write_gamma(offset as u64),   // TODO: does the param have to be usize?
            EncodingType::DELTA => offset_obs.write_delta(offset as u64),
            other => return Err(format!("The {} offset coding is not supported", other)),
        };
        Ok(offset)
    }

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
                if j >= self.parameters.min_interval_len {
                    left.push(v[i]);
                    len.push(j);
                    n_interval += 1;
                    i += j - 1;
                }
            }

            if j < self.parameters.min_interval_len {
                residuals.push(v[i]);
            }

            i += 1;
        }

        n_interval
    }
    
    fn diff_comp(
        &self,
        graph_obs: &mut OutputBitStreamBuilder,
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
        if self.parameters.window_size > 0 {
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

            if self.parameters.min_interval_len != 0 {
                // If we are to produce intervals, we first compute them
                let interval_count = self.intervalize(&extras, &mut left, &mut len, &mut residuals);

                _t = graph_obs.write_gamma(interval_count as u64) as usize;

                let mut curr_int_len;

                for i in 0..interval_count {
                    if i == 0 {
                        prev = left[i];
                        _t = graph_obs.write_gamma(self.int2nat(prev as i64 - curr_node as i64)) as usize;
                    } else {
                        _t = graph_obs.write_gamma((left[i] - prev - 1) as u64) as usize;
                    }
                    
                    curr_int_len = len[i];
                    
                    prev = left[i] + curr_int_len;
                    
                    _t = graph_obs.write_gamma((curr_int_len - self.parameters.min_interval_len) as u64) as usize;
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
                _t = self.write_residual(graph_obs, self.int2nat(prev as i64 - curr_node as i64) as usize)?;
                for i in 1..residual_count {
                    if residual[i] == prev {
                        return Err(format!("Repeated successor {} in successor list of node {}", prev, curr_node));
                    }
                    _t = self.write_residual(graph_obs, residual[i] - prev - 1)?;
                    prev = residual[i];
                }
            }
        }

        Ok(graph_obs.written_bits - written_data_at_start)
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
    //         parameters: BVGraphParameters { 
    //             max_ref_count, 
    //             window_size, 
    //             min_interval_len, 
    //             zeta_k, 
    //             outdegree_coding: EncodingType::GAMMA, 
    //             block_coding: EncodingType::GAMMA, 
    //             residual_coding: EncodingType::ZETA, 
    //             reference_coding: EncodingType::UNARY, 
    //             block_count_coding: EncodingType::GAMMA, 
    //             offset_coding: EncodingType::GAMMA 
    //         }
    //     })
    // }

    fn int2nat(&self, x: i64) -> u64 {
        ((x << 1) ^ (x >> (i64::BITS - 1))) as u64
    } 
}

pub struct BVGraphBuilder {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<u8>,   // TODO: does this BVGraph implementation have to deal with generics instead of bytes?
    loaded_offsets: Vec<u8>,
    cached_node: Option<usize>,
    cached_outdegree: Option<usize>,
    cached_ptr: Option<usize>,
    parameters: BVGraphParameters,
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
        let mut graph_with_outdegrees = Vec::default();
        graph_with_outdegrees.reserve(graph.graph_memory.len());

        let mut n = 0;

        for (i, x) in graph.graph_memory.iter().enumerate() {
            graph_with_outdegrees.push(x.to_u8().unwrap());
            if n == 0 || graph.offsets[n - 1] == i {
                let outd = graph.outdegree_internal(n.into());
                graph_with_outdegrees.push(outd as u8);
                n += 1;
            }
        }

        let mut new_offsets = Vec::default();
        new_offsets.reserve(graph.offsets.len());

        let mut to_add = 1;
        for x in graph.offsets.iter() {
            new_offsets.push(x.to_u8().unwrap() + to_add);
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
            parameters: BVGraphParameters { 
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
}

impl Default for BVGraphBuilder {
    fn default() -> Self {
        BVGraphBuilder { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Vec::default(), 
            loaded_offsets: Vec::default(), 
            cached_node: None, 
            cached_outdegree: None, 
            cached_ptr: None,
            parameters: BVGraphParameters {
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
}

impl BVGraphBuilder {
    pub fn new() -> BVGraphBuilder {
        Self::default()
    }

    pub fn load_graph(mut self, filename: &str) -> Self {
        let props = serde_json::from_str::<Properties>(
                        fs::read_to_string(format!("{}.properties", filename)).unwrap().as_str()
                    ).unwrap();
        
        let n = props.nodes;
        if n as u64 > u64::MAX {
            panic!("This version of WebGraph cannot handle graphs with {} (>=2^63) nodes", n);
        }
        let m = props.arcs;
        let window_size = props.window_size;
        let max_ref_count = props.max_ref_count;
        let min_interval_len = props.min_interval_len;
        let zeta_k = props.zeta_k;  // TODO: manage absence

        let graph_ibs = bincode::deserialize::<Box<[u8]>>(
                                    fs::read(format!("{}.graph", filename)).unwrap().as_slice()
                                   ).unwrap();
        let offsets_ibs = bincode::deserialize::<Box<[u8]>>(
                                    fs::read(format!("{}.offsets", filename)).unwrap().as_slice()
                                     ).unwrap();

        self.loaded_graph = graph_ibs.to_vec();
        self.loaded_offsets = offsets_ibs.to_vec();

        self
    }

    /// Loads a graph file represented in plain mode.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the compressed graph file
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
    /// * `filename` - The filename of the compressed offsets file
    pub fn load_offsets_plain(mut self, filename: &str) -> Self {

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

    /// Sets the maximum reference chain length.
    /// 
    /// # Arguments
    /// 
    /// * `ref_count` - The maximum length of the chain.
    pub fn set_ref_count(mut self, ref_count: usize) -> Self {
        self.parameters.max_ref_count = ref_count;

        self
    }

    /// Sets the maximum reference window size.
    /// 
    /// # Arguments
    /// 
    /// * `window_size` - The maximum length of the window.
    pub fn set_window_size(mut self, window_size: usize) -> Self {
        self.parameters.window_size = window_size;

        self
    }

    /// Sets the minimum length of the intervals.
    /// 
    /// # Arguments
    /// 
    /// * `min_interval_length` - The minimum length of the intervals.
    pub fn set_min_interval_len(mut self, min_interval_len: usize) -> Self {
        self.parameters.min_interval_len = min_interval_len;

        self
    }

    /// Sets the *k* parameter for *zeta*-coding.
    /// 
    /// # Arguments
    /// 
    /// * `zeta_k` - The *k* parameter.
    pub fn set_zeta_k(mut self, zeta_k: usize) -> Self {
        self.parameters.zeta_k = zeta_k;

        self
    }

    /// Sets the encoding type for outdegrees.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_outdegree_coding(mut self, code: EncodingType) -> Self {
        self.parameters.outdegree_coding = code;

        self
    }
    
    /// Sets the encoding type for blocks.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_block_coding(mut self, code: EncodingType) -> Self {
        self.parameters.block_coding = code;

        self
    }
    
    /// Sets the encoding type for the lists of residuals.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_residual_coding(mut self, code: EncodingType) -> Self {
        self.parameters.residual_coding = code;

        self
    }

    /// Sets the encoding type for references.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_reference_coding(mut self, code: EncodingType) -> Self {
        self.parameters.reference_coding = code;

        self
    }

    /// Sets the encoding type for the block counters.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_block_count_coding(mut self, code: EncodingType) -> Self {
        self.parameters.block_count_coding = code;

        self
    }

    /// Sets the encoding type for the offsets.
    /// 
    /// # Arguments
    /// 
    /// * `code` - The encoding type.
    pub fn set_offset_coding(mut self, code: EncodingType) -> Self {
        self.parameters.offset_coding = code;

        self
    }

    /// Constructs the BVGraph object.
    pub fn build(self) -> BVGraph {
        BVGraph { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph.into_boxed_slice(), 
            offsets: self.loaded_offsets.into_boxed_slice(), 
            cached_node: self.cached_node, 
            cached_outdegree: self.cached_outdegree, 
            cached_ptr: self.cached_ptr,
            parameters: BVGraphParameters {
                max_ref_count: self.parameters.max_ref_count, 
                window_size: self.parameters.window_size, 
                min_interval_len: self.parameters.min_interval_len, 
                zeta_k: self.parameters.zeta_k, 
                outdegree_coding: self.parameters.outdegree_coding, 
                block_coding: self.parameters.block_coding, 
                residual_coding: self.parameters.residual_coding, 
                reference_coding: self.parameters.reference_coding, 
                block_count_coding: self.parameters.block_count_coding, 
                offset_coding: self.parameters.offset_coding
            }
        }
    }
}