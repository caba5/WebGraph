pub mod webgraph;
pub mod bitstreams;
pub mod ascii_graph;
pub mod properties;
pub mod utils;
pub mod huffman_zuckerli;

pub trait ImmutableGraph {
    type NodeT;

    fn num_nodes(&self) -> usize;
    fn num_arcs(&self) -> usize;
    fn outdegree(&self, x: Self::NodeT) -> Option<usize>; 
    fn successors(&self, x: Self::NodeT) -> Box<[Self::NodeT]>;
    fn store(&mut self, filename: &str) -> std::io::Result<()>;
}