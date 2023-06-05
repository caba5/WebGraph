use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub mod webgraph;
pub mod bitstream;
pub mod uncompressed_graph;

pub trait ImmutableGraph {
    type NodeT;

    fn num_nodes(&self) -> usize;
    fn num_arcs(&self) -> usize;
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize>; 
    // fn successors(&self, x: Self::NodeT) -> Box<dyn Iterator<Item = &u32>>; // TODO: remove dyn, use a specific iterator type
    // fn node_iterator(&self) -> iter;
    // fn outdegrees(&self) -> iter;
    fn store(&self, filename: &str) -> std::io::Result<()>;
    // fn load(filename: &str) -> Self;
}

#[derive(Serialize, Deserialize)]
struct Properties {
    nodes: usize,
    tot_links: usize,
    arcs: usize,
    window_size: usize,
    max_ref_count: usize,
    min_interval_len: usize,
    zetak: usize,
    comp: EncodingType,
    avg_ref: f32,
    avg_dist: f32,
    copied_arcs: usize,
    intervalized_arcs: usize,
    residual_arcs: usize,
    bits_per_link: f32,
    comp_ratio: f32,
    bits_per_node: f32,
    avg_bits_for_outdeg: f32,
    avg_bits_for_refs: f32,
    avg_bits_for_blocks: f32,
    avg_bits_for_residuals: f32,
    avg_bits_for_intervals: f32,
    bits_for_outdeg: usize,
    bits_for_refs: usize,
    bits_for_blocks: usize,
    bits_for_residuals: usize,
    bits_for_intervals: usize,
}

#[derive(Clone, Eq, PartialEq, Serialize, Deserialize, Debug, ValueEnum)]
pub enum EncodingType {  
    GAMMA,
    DELTA,
    ZETA,
    NIBBLE,
    GOLOMB,
    SKEWEDGOLOMB,
    UNARY,
}