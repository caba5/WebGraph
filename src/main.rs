use webgraph_rust::{webgraph::BVGraph, EncodingType};

use clap::Parser;

#[derive(Parser, Debug)]
struct WGArgs {
    /// The compression method to use
    compression_method: EncodingType,
    /// The size of the window
    window_size: usize,
    /// The maximum reference chain length
    max_ref_count: usize,
    /// The minimum length of the interval
    min_interval_length: usize,
    /// The *k* parameter in zeta_k encoding
    zetak: usize,
    /// Source filename
    source_name: String,
    /// Destination filename
    dest_name: String,
}

fn main() {

    let args = WGArgs::parse();

    let graph = BVGraph::<u32>::load(&args.source_name); // Loading the graph

    // ...ops...

    graph.store(&args.dest_name); // Storing the graph

}