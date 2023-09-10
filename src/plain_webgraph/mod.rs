use std::{fmt::Display, marker::PhantomData, fs, cmp::Ordering, str::FromStr};

use serde::{Serialize, Deserialize};

use crate::{ImmutableGraph, Properties, nat2int, int2nat, EncodingType, uncompressed_graph::UncompressedGraph};

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub struct BVGraphPlain {
    n: usize,
    m: usize,
    /// Each list has the following structure: \[node, outdegree, successors_list\].
    pub graph_memory: Box<[usize]>,
    pub offsets: Box<[usize]>,
    cached_node: Option<usize>,
    cached_outdegree: Option<usize>,
    cached_ptr: Option<usize>,
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: Option<u64>,
}

impl ImmutableGraph for BVGraphPlain {
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

        let mut node_iter = BVGraphPlainIterator {
            curr: 0, 
            graph: self.as_ref(),
        };
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1]}).ok()?;

        self.cached_outdegree = 
            if let Some(outd) = self.read_outdegree(&mut node_iter) {
                Some(outd)
            } else {
                None
            };

        self.cached_outdegree
    }

    /// Stores the graph as the Webgraph algorithm would but in plain (string) mode 
    /// and without any encoding type.
    fn store(&mut self, filename: &str) -> std::io::Result<()> {
        let mut bit_offset = 0;

        let mut graph_buf = Vec::default();
        let mut offsets_buf = Vec::default();

        let cyclic_buff_size = self.window_size + 1;
        // Cyclic array of previous lists
        let mut list = vec![vec![0; 1024]; cyclic_buff_size];
        // The length of each list
        let mut list_len = vec![0; cyclic_buff_size];
        // The depth of the references of each list
        let mut ref_count: Vec<i32> = vec![0; cyclic_buff_size];

        let mut node_iter = self.iter();
        
        while node_iter.has_next() {
            let curr_node = node_iter.next().unwrap();
            let outd = node_iter.next().unwrap();
            let curr_idx = curr_node % cyclic_buff_size;
            
            println!("Curr node: {}", curr_node);
            
            offsets_buf.push(graph_buf.len() - bit_offset);
            
            bit_offset = graph_buf.len();

            graph_buf.push(outd);

            if outd > list[curr_idx].len() {
                list[curr_idx].resize(outd, 0);
            }
            
            let mut successors = Vec::default();
            let mut successors_it = self.successors_plain(curr_node).unwrap();

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
                    if ref_count[cand as usize] < (self.max_ref_count as i32) && list_len[cand as usize] != 0 {
                        let diff_comp = 
                            self.diff_comp(&mut graph_buf, 
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
                    &mut graph_buf, 
                    curr_node, 
                    best_ref as usize, 
                    list[best_cand as usize].as_slice(), 
                    list[curr_idx].as_slice(),
                ).unwrap();
            }
            
            node_iter.advance_by(outd).unwrap();
        }
        
        // We write the final offset to the offset stream
        offsets_buf.push(graph_buf.len() - bit_offset);
        
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
        
        let graph_buf: Vec<String> = graph_buf.into_iter().map(|val| format!("{} ", val)).collect();
        let graph_buf = graph_buf.concat();
        
        fs::write(format!("{}.graph", filename), graph_buf)?;

        let offsets_buf: Vec<String> = offsets_buf.into_iter().map(|val| format!("{} ", val)).collect();
        let offsets_buf = offsets_buf.concat();

        fs::write(format!("{}.offsets", filename), offsets_buf)?;

        let props = Properties {
            nodes: self.n,
            arcs: self.m,
            window_size: self.window_size,
            max_ref_count: self.max_ref_count,
            min_interval_len: self.min_interval_len,
            zeta_k: self.zeta_k,
            ..Default::default()
        };

        fs::write(format!("{}.properties", filename), serde_json::to_string(&props).unwrap())?;

        Ok(())
    }
}

/// Defines an iterator over all the elements of the graph vector.
pub struct BVGraphPlainIterator<BV: AsRef<BVGraphPlain>> {
    curr: usize,
    graph: BV,
}

impl<BV: AsRef<BVGraphPlain>> Iterator for BVGraphPlainIterator<BV> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr > self.graph.as_ref().graph_memory.len() {
            return None;
        }

        let res = Some(self.graph.as_ref().graph_memory[self.curr]);

        self.curr += 1;

        res
    }
}

impl<BV: AsRef<BVGraphPlain>> BVGraphPlainIterator<BV> {
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
    /// Returns [`Ok(())`][Ok] if the advancing operation was successful or an error otherwise.
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
pub struct BVGraphPlainSuccessorsIterator<BV: AsRef<BVGraphPlain>> {
    base: usize,
    idx_from_base: usize,
    up_to: usize,
    graph: BV,
}

impl<BV: AsRef<BVGraphPlain>> Iterator for BVGraphPlainSuccessorsIterator<BV> {
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

impl<BV: AsRef<BVGraphPlain>> BVGraphPlainSuccessorsIterator<BV> {
    /// Returns `true` if the iterator has not reached the final successor of the node, `false` otherwise.
    #[inline(always)]
    fn has_next(&self) -> bool {
        self.base + self.idx_from_base + 1 < self.up_to
    }
}

impl AsMut<BVGraphPlain> for BVGraphPlain {

    fn as_mut(&mut self) -> &mut BVGraphPlain {
        self
    }
}

impl AsRef<BVGraphPlain> for BVGraphPlain {
    fn as_ref(&self) -> &BVGraphPlain {
        self
    }
}

impl IntoIterator for BVGraphPlain
{
    type Item = usize;

    type IntoIter = BVGraphPlainIterator<BVGraphPlain>;

    fn into_iter(self) -> Self::IntoIter {
        BVGraphPlainIterator {
            curr: 0,
            graph: self,
        }
    }
}

impl BVGraphPlain {
    pub fn iter(&self) -> BVGraphPlainIterator<&Self> {
        BVGraphPlainIterator {
            curr: 0,
            graph: self,
        }
    }

    // fn outdegree_internal(&self, x: usize, decoder: &mut BinaryReader) -> usize { // TODO: reintroduce mut and caches
    //     if self.cached_node.is_some() && x == self.cached_node.unwrap() {
    //         return self.cached_outdegree.unwrap();
    //     }
        
    //     decoder.position(self.offsets[x] as u64); // TODO: offsets are encoded
    //     // self.cached_node = Some(x);
    //     let d = OutdegreeCoding::read_next(decoder, self.zeta_k) as usize;
    //     // self.cached_outdegree = Some(d);
    //     // self.cached_ptr = Some(decoder.position);
    //     d
    // }

    // fn decode_list(&self, x: usize, decoder: &mut BinaryReader, window: Option<&mut Vec<Vec<usize>>>, outd: &mut [usize]) -> Box<[usize]> {
    //     let cyclic_buffer_size = self.window_size + 1;
    //     let degree;
    //     if window.is_none() {
    //         degree = self.outdegree_internal(x, decoder);
    //     } else {
    //         degree = OutdegreeCoding::read_next(decoder, self.zeta_k) as usize;
    //         outd[x % cyclic_buffer_size] = degree; 
    //     }

    //     if degree == 0 {
    //         return Box::new([]);
    //     }

    //     let mut reference = -1;
    //     if self.window_size > 0 {
    //         reference = ReferenceCoding::read_next(decoder, self.zeta_k) as i64;
    //     }

    //     // Position in the circular buffer of the reference of the current node
    //     let reference_index = ((x as i64 - reference + cyclic_buffer_size as i64) as usize) % cyclic_buffer_size;

    //     let mut block = Vec::default();

    //     let mut extra_count;

    //     if reference > 0 {
    //         let block_count = BlockCountCoding::read_next(decoder, self.zeta_k) as usize;
    //         if block_count != 0 {
    //             block = Vec::with_capacity(block_count);
    //         }

    //         let mut copied = 0; // # of copied successors
    //         let mut total = 0; // total # of successors specified in some copy block

    //         let mut i = 0;
    //         while i < block_count {
    //             block.push(BlockCoding::read_next(decoder, self.zeta_k) as usize + if i == 0 {0} else {1});
    //             total += block[i];
    //             if (i & 1) == 0 { // Alternate, count only even blocks
    //                 copied += block[i];
    //             }

    //             i += 1;
    //         }

    //         // If the block count is even, we must compute the number of successors copied implicitly
    //         if (block_count & 1) == 0 {
    //             copied += (
    //                 if window.is_some() {outd[reference_index]} 
    //                 else {self.outdegree_internal((x as i64 - reference) as usize, &mut decoder.clone())}
    //             ) - total;
    //         }
            
    //         extra_count = degree - copied;
    //     } else {
    //         extra_count = degree;
    //     }

    //     let mut interval_count = 0; // Number of intervals

    //     let mut left = Vec::default();
    //     let mut len = Vec::default();

    //     if extra_count > 0 && self.min_interval_len != 0 {
    //         interval_count = GammaCode::read_next(decoder, self.zeta_k) as usize;
            
    //         if interval_count != 0 {
    //             left = Vec::with_capacity(interval_count);
    //             len = Vec::with_capacity(interval_count);

    //             left.push(nat2int(GammaCode::read_next(decoder, self.zeta_k)) + x as i64);
    //             len.push(GammaCode::read_next(decoder, self.zeta_k) as usize + self.min_interval_len);
    //             let mut prev = left[0] + len[0] as i64;  // Holds the last integer in the last interval
    //             extra_count -= len[0];

    //             let mut i = 1;
    //             while i < interval_count {
    //                 prev += GammaCode::read_next(decoder, self.zeta_k) as i64 + 1;

    //                 left.push(prev);
    //                 len.push(GammaCode::read_next(decoder, self.zeta_k) as usize + self.min_interval_len);

    //                 prev += len[i] as i64;
    //                 extra_count -= len[i];

    //                 i += 1;
    //             }
    //         }
    //     }

    //     let mut residual_list = Vec::with_capacity(extra_count);
    //     if extra_count > 0 {
    //         residual_list.push(x as i64 + nat2int(ResidualCoding::read_next(decoder, self.zeta_k)));
    //         let mut remaining = extra_count - 1;
    //         let mut curr_len = 1;

    //         while remaining > 0 {
    //             residual_list.push(residual_list[curr_len - 1] + ResidualCoding::read_next(decoder, self.zeta_k) as i64 + 1);
    //             curr_len += 1;

    //             remaining -= 1;
    //         }
    //     }

    //     // The extra part is made by the contribution of intervals, if any, and by the residuals list.
    //     let mut extra_list = if interval_count == 0 {residual_list.clone()} else {Vec::default()};
    //     if interval_count > 0 {
    //         let mut curr_left = if !left.is_empty() {left[0]} else {0};
    //         let mut curr_index = 0;
    //         let mut curr_interval = 0;
    //         let mut remaining = left.len();

    //         while remaining > 0 {
    //             extra_list.push(curr_left + curr_index as i64);
    //             curr_index += 1;

    //             if curr_index == len[curr_interval] {
    //                 remaining -= 1;
    //                 if remaining != 0 {
    //                     curr_interval += 1;
    //                     curr_left = left[curr_interval];
    //                 }
    //                 curr_index = 0;
    //             }
    //         }
    //     } 
    //     if interval_count > 0 && extra_count > 0 {
    //         let len_residual = residual_list.len();
    //         let len_extra = extra_list.len();

    //         let mut temp_list = Vec::with_capacity(len_residual + len_extra);
    //         let mut idx0 = 0;
    //         let mut idx1 = 0;
    //         while idx0 < len_residual && idx1 < len_extra {
    //             if residual_list[idx0] <= extra_list[idx1] {
    //                 temp_list.push(residual_list[idx0]);
    //                 idx0 += 1;
    //             } else {
    //                 temp_list.push(extra_list[idx1]);
    //                 idx1 += 1;
    //             }
    //         }

    //         while idx0 < len_residual {
    //             temp_list.push(residual_list[idx0]);
    //             idx0 += 1;
    //         }

    //         while idx1 < len_extra {
    //             temp_list.push(extra_list[idx1]);
    //             idx1 += 1;
    //         }

    //         extra_list = temp_list;
    //     }

    //     let mut block_list = Vec::default();
    //     if reference > 0 {
    //         let decoded_reference;

    //         let mut reference_it = 
    //             if let Some(window) = window {
    //                 window[reference_index][0..outd[reference_index]].iter()
    //             } else {
    //                 decoded_reference = self.decode_list(
    //                     (x as i64 - reference) as usize, 
    //                     &mut self.graph_memory, 
    //                     None, 
    //                     &mut []
    //                 );
    //                 decoded_reference.iter()
    //             };
            
    //         let mask_len = block.len();
    //         let mut curr_mask = 0;
    //         let mut left;

    //         if mask_len != 0 {
    //             left = block[curr_mask] as i64;
    //             curr_mask += 1;
    //             if left == 0 && curr_mask < mask_len {
    //                 reference_it.nth(block[curr_mask] - 1);
    //                 curr_mask += 1;

    //                 left = if curr_mask < mask_len {curr_mask += 1; block[curr_mask - 1] as i64} else {-1};
    //             }
    //         } else {
    //             left = -1;
    //         }

    //         while left != 0 {
    //             let next = reference_it.next();

    //             if next.is_none() {
    //                 break;
    //             }

    //             if left == -1 {
    //                 block_list.push(*next.unwrap());
    //             }
                
    //             if left > 0 {
    //                 left -= 1;
    //                 if left == 0 && curr_mask < mask_len {
    //                     reference_it.nth(block[curr_mask] - 1);
    //                     curr_mask += 1;

    //                     left = if curr_mask < mask_len {curr_mask += 1; block[curr_mask - 1] as i64} else {-1};
    //                 }
    //                 block_list.push(*next.unwrap());
    //             }
                
    //         }
    //     }

    //     if reference <= 0 {
    //         let extra_list: Vec<usize> = extra_list.iter().map(|x| x.to_usize().unwrap()).collect();
    //         return extra_list.into_boxed_slice();
    //     } else if extra_list.is_empty() {
    //         return block_list.into_boxed_slice();
    //     };

    //     let len_block = block_list.len();
    //     let len_extra = extra_list.len();

    //     let mut temp_list = Vec::with_capacity(len_block + len_extra);
    //     let mut idx0 = 0;
    //     let mut idx1 = 0;
    //     while idx0 < len_block && idx1 < len_extra {
    //         if block_list[idx0] < extra_list[idx1] as usize {
    //             temp_list.push(block_list[idx0]);
    //             idx0 += 1;
    //         } else {
    //             temp_list.push(extra_list[idx1] as usize);
    //             idx1 += 1;
    //         }
    //     }

    //     while idx0 < len_block {
    //         temp_list.push(block_list[idx0]);
    //         idx0 += 1;
    //     }

    //     while idx1 < len_extra {
    //         temp_list.push(extra_list[idx1] as usize);
    //         idx1 += 1;
    //     }

    //     temp_list.into_boxed_slice()
    // }
    
    fn outdegree_internal_plain(&self, x: usize) -> usize {
        let mut node_iter = BVGraphPlainIterator {
            curr: 0, 
            graph: self,
        };
        node_iter.position_to(if x == 0 {0} else {self.offsets[x - 1]}).ok();
        self.read_outdegree(&mut node_iter).unwrap()
    }
    
    fn successors_plain(&self, x: usize) -> Option<BVGraphPlainSuccessorsIterator<&BVGraphPlain>>
    {
        if x > self.n - 1 {
            return None;
        }

        let base = if x == 0 {0} else { self.offsets[x - 1] };
        Some(BVGraphPlainSuccessorsIterator {
            base,
            idx_from_base: 1, // starts from the outdeg
            up_to: base + self.outdegree_internal_plain(x) + 2, // summing 2 to skip the node and its outdeg
            graph: self,
        })
    }

    #[inline(always)]
    fn read_outdegree(
        &self, 
        outdegree_iter: &mut BVGraphPlainIterator<&Self>
    ) -> Option<usize> {
        outdegree_iter.next()
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
        graph_obs: &mut Vec<usize>,
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

        let written_data_at_start = graph_obs.len();

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
            graph_obs.push(reference);
        }

        // Then, if the reference is not void we write the length of the copy list
        if reference != 0 {
            graph_obs.push(block_count);

            // Then, we write the copy list; all lengths except the first one are decremented
            if block_count > 0 {
                graph_obs.push(blocks[0]);
                for blk in blocks.iter().skip(1) {
                    graph_obs.push(blk - 1);
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

                graph_obs.push(interval_count);

                let mut curr_int_len;

                for i in 0..interval_count {
                    if i == 0 {
                        prev = left[i];
                        graph_obs.push(int2nat(prev as i64 - curr_node as i64) as usize);
                    } else {
                        graph_obs.push(left[i] - prev - 1);
                    }
                    
                    curr_int_len = len[i];
                    
                    prev = left[i] + curr_int_len;
                    
                    graph_obs.push(curr_int_len - self.min_interval_len);
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
                graph_obs.push(int2nat(prev as i64 - curr_node as i64) as usize);
                for i in 1..residual_count {
                    if residual[i] == prev {
                        return Err(format!("Repeated successor {} in successor list of node {}", prev, curr_node));
                    }
                    
                    graph_obs.push(residual[i] - prev - 1);
                    prev = residual[i];
                }
            }
        }

        Ok(graph_obs.len() - written_data_at_start)
    }
}

pub struct BVGraphPlainBuilder {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Box<[usize]>,
    loaded_offsets: Box<[usize]>,
    cached_node: Option<usize>,
    cached_outdegree: Option<usize>,
    cached_ptr: Option<usize>,
    max_ref_count: usize,
    window_size: usize,
    min_interval_len: usize,
    zeta_k: Option<u64>, 
    outdegree_coding: EncodingType, 
    block_coding: EncodingType, 
    residual_coding: EncodingType, 
    reference_coding: EncodingType, 
    block_count_coding: EncodingType, 
    offset_coding: EncodingType,
}

impl<T> From<UncompressedGraph<T>> for BVGraphPlainBuilder 
where T:
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
        + Clone
        + From<usize>
{
    fn from(graph: UncompressedGraph<T>) -> Self {
        let mut graph_with_outdegrees = Vec::with_capacity(
            graph.graph_memory.len() +  // All of the graph's elements
            graph.num_nodes() * 2       // Each node's outdegree
        );

        let mut n = 0;

        for (i, x) in graph.graph_memory.iter().enumerate() {
            graph_with_outdegrees.push(x.to_usize().unwrap());
            if n == 0 || graph.offsets[n - 1] == i { // Catch the beginning of a node's list
                let outd = graph.outdegree_internal(n.into());
                graph_with_outdegrees.push(outd);
                n += 1;
            }
        }

        let mut new_offsets = Vec::with_capacity(graph.offsets.len());

        let mut to_add = 1; // Fixes the additional offset generated by adding the outdegree to each node
        for x in graph.offsets.iter() {
            new_offsets.push(x + to_add);
            to_add += 1;
        }

        Self { 
            num_nodes: graph.num_nodes(), 
            num_edges: graph.num_arcs(),
            loaded_graph: graph_with_outdegrees.into_boxed_slice(),
            loaded_offsets: new_offsets.into_boxed_slice(),
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
        }
    }
}

impl Default for BVGraphPlainBuilder {
    fn default() -> Self {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Box::default(), 
            loaded_offsets: Box::default(),
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
        }
    }
}

impl BVGraphPlainBuilder {
    pub fn new() -> BVGraphPlainBuilder {
        Self::default()
    }

    /// Loads an uncompressed graph file represented in plain mode.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The basename of the compressed graph file
    pub fn load_graph_uncompressed(mut self, filename: &str) -> Self {
        self.loaded_graph = fs::read_to_string(format!("{}.graph.plain", filename))
                            .expect("Failed to load the graph file")
                            .split(' ')
                            .map(|node| node
                                                .parse()
                                                // This should also account for overflows
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
    pub fn load_offsets_uncompressed(mut self, filename: &str) -> Self {

        self.loaded_offsets = fs::read_to_string(format!("{}.offsets.plain", filename))
                            .expect("Failed to load the offsets file")
                            .split(' ')
                            .map(|node| node
                                                .parse()
                                                .unwrap_or_else(|_| panic!("Failed to parse offset {}", node))
                            )
                            .collect();

        self
    }

    /// Sets the number of nodes of the graph.
    pub fn count_nodes(mut self) -> Self {
        assert!(self.loaded_offsets.len() > 0, "The offsets file has to be loaded first.");

        self.num_nodes = self.loaded_offsets.len();

        self
    }

    /// Sets the number of edges of the graph.
    pub fn count_edges(mut self) -> Self {
        assert!(self.loaded_graph.len() > 0, "The graph file has to be loaded first.");
        assert!(self.loaded_offsets.len() > 0, "The offsets file has to be loaded first.");

        self.num_edges = self.loaded_graph.len() - (self.loaded_offsets.len() * 2);

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
    pub fn build(self) -> BVGraphPlain {
        BVGraphPlain { 
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
        }
    }
}

#[cfg(test)]
mod tests;