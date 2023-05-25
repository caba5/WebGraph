use serde::{Deserialize, Serialize};

pub mod webgraph;
pub mod bitstream;


pub trait ImmutableGraph {
    type NodeT;

    fn num_nodes(&self) -> usize;
    fn num_arcs(&self) -> usize;
    fn outdegree(&mut self, x: Self::NodeT) -> Option<u32>; 
    fn successors(&self, x: u32) -> Result<Box<dyn Iterator<Item = &u32>>, &str>; // TODO: remove dyn, use a specific iterator type
    // fn node_iterator(&self) -> iter;
    // fn outdegrees(&self) -> iter;
    // TODO: how to use fn store here and solve the problem of serializing 'self'
}

#[derive(Serialize, Deserialize)]
struct Properties {
    nodes: u32,
    tot_links: u32,
    arcs: u32,
    window_size: u32,
    max_ref_count: u32,
    min_interval_len: u32,
    zetak: u32,
    comp: EncodingType,
    avg_ref: f32,
    avg_dist: f32,
    copied_arcs: u32,
    intervalized_arcs: u32,
    residual_arcs: u32,
    bits_per_link: f32,
    comp_ratio: f32,
    bits_per_node: f32,
    avg_bits_for_outdeg: f32,
    avg_bits_for_refs: f32,
    avg_bits_for_blocks: f32,
    avg_bits_for_residuals: f32,
    avg_bits_for_intervals: f32,
    bits_for_outdeg: u32,
    bits_for_refs: u32,
    bits_for_blocks: u32,
    bits_for_residuals: u32,
    bits_for_intervals: u32,
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug)]
pub enum EncodingType {  
    GAMMA,
    DELTA,
    ZETA,
    NIBBLE,
    GOLOMB,
    SKEWEDGOLOMB, // Taken from CompressionFlags.java
    UNARY,
}