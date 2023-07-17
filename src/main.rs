use webgraph_rust::{webgraph::BVGraphBuilder, EncodingType, ImmutableGraph};

use clap::Parser;

#[derive(Parser, Debug)]
struct WGArgs { // TODO: automatic compression methods
    /// The compression method to use
    compression_method: EncodingType,
    /// The size of the window
    window_size: usize,
    /// The maximum reference chain length
    max_ref_count: usize,
    /// The minimum length of the interval
    min_interval_length: usize,
    /// The *k* parameter in zeta_k encoding
    zeta_k: usize,
    /// Source filename
    source_name: String,
    /// Destination filename
    dest_name: String,
}

fn main() {

    let args = WGArgs::parse();

    let graph = BVGraphBuilder::new()
                .load_graph(&args.source_name)
                .load_offsets(&args.source_name)
                .set_window_size(args.window_size)
                .set_ref_count(args.max_ref_count)
                .set_min_interval_len(args.min_interval_length)
                .set_zeta_k(args.zeta_k)
                .construct();

    graph.store(&args.dest_name).expect("Failed storing the graph");
}