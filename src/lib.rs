use std::fmt::Display;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub mod webgraph;
pub mod bitstreams;
pub mod uncompressed_graph;

pub trait ImmutableGraph {
    type NodeT;

    fn num_nodes(&self) -> usize;
    fn num_arcs(&self) -> usize;
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize>; 
    // fn successors(&self, x: Self::NodeT) -> Box<dyn Iterator<Item = &u32>>; // TODO: remove dyn, use a specific iterator type
    // fn node_iterator(&self) -> iter;
    fn store(&self, filename: &str) -> std::io::Result<()>;
    // fn load(filename: &str) -> Self;
}

#[derive(Serialize, Deserialize, Default)]
struct Properties {
    nodes: usize,
    arcs: usize,
    window_size: usize,
    max_ref_count: usize,
    min_interval_len: usize,
    zeta_k: usize,
    outdegree_coding: EncodingType,
    block_coding: EncodingType,
    residual_coding: EncodingType,
    reference_coding: EncodingType,
    block_count_coding: EncodingType,
    offset_coding: EncodingType,
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

#[derive(Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Debug, ValueEnum, Default)]
pub enum EncodingType {
    #[default]
    GAMMA,
    DELTA,
    ZETA,
    NIBBLE,
    GOLOMB,
    SKEWEDGOLOMB,
    UNARY,
}

impl Display for EncodingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            EncodingType::GAMMA => "gamma",
            EncodingType::DELTA => "delta",
            EncodingType::ZETA => "zeta",
            EncodingType::NIBBLE => "nibble",
            EncodingType::GOLOMB => "golomb",
            EncodingType::SKEWEDGOLOMB => "skewed golomb",
            EncodingType::UNARY => "unary"
        })
    }
}