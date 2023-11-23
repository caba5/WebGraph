use std::{time::Instant, fs::File, io::BufReader};

use clap::Parser;
use rand::Rng;
use webgraph_rust::{properties::Properties, webgraph::zuckerli_in::{BVGraphBuilder, NUM_CONTEXTS}, utils::{encodings::{GammaCode, UnaryCode, ZetaCode, Huff}, EncodingType}, ImmutableGraph, huffman_zuckerli::huffman_decoder::HuffmanDecoder, bitstreams::BinaryReader};

const N_QUERIES: usize = 1000000;

#[derive(Parser, Debug)]
#[command(about = "Generate a graph having the blocks, the intervals and the residuals Huffman-encoded")]
struct Args {
    /// The size of the window
    #[arg(short = 'w', long = "window-size", default_value_t = 7)]
    window_size: usize,
    /// The maximum reference chain length
    #[arg(short = 'r', long = "max-ref-count", default_value_t = 3)]
    max_ref_count: usize,
    /// The minimum length of the interval
    #[arg(short = 'i', long = "min-interval-len", default_value_t = 4)]
    min_interval_length: usize,
    /// Specifies the k parameter for ZetaK coding
    #[arg(short = 'k', long = "zetak", default_value_t = 3)]
    zeta_k: usize,
    /// The basename of the huffman-compressed graph file
    source_name: String,
    /// The destination basename of the graph file
    dest_name: Option<String>,
    /// Performance test
    #[arg(short, long = "perf", default_value_t = false)]
    perf_test: bool,
}

fn main() {
    let args = Args::parse();

    if !args.perf_test && args.dest_name.is_none() {
        panic!("No destination name provided");
    }

    let properties_file = File::open(format!("{}.properties", args.source_name));
    let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
    let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));
    let props = Properties::from(p);
    
    match (props.block_coding, props.block_count_coding, props.outdegree_coding, props.offset_coding, props.reference_coding, props.interval_coding, props.residual_coding) {
        (EncodingType::HUFFMAN, EncodingType::GAMMA, EncodingType::HUFFMAN, EncodingType::GAMMA, EncodingType::UNARY, EncodingType::HUFFMAN, EncodingType::HUFFMAN) => {},
        _ => panic!("The graph has to be Huffman-encoded (that is, its blocks, intervals, and residuals must be encoded by Huffman)")
    };

    let mut bvgraph = BVGraphBuilder::<
        Huff, GammaCode, Huff, GammaCode, UnaryCode, Huff, Huff,
        // Default encoding
        GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode
    >::new()
        .set_in_min_interval_len(props.min_interval_len)
        .set_out_min_interval_len(args.min_interval_length)
        .set_in_max_ref_count(props.max_ref_count)
        .set_out_max_ref_count(args.max_ref_count)
        .set_in_window_size(props.window_size)
        .set_out_window_size(args.window_size)
        .set_in_zeta(props.zeta_k)
        .set_out_zeta(Some(args.zeta_k as u64))
        .set_num_nodes(props.nodes)
        .set_num_edges(props.arcs)
        .load_graph(&args.source_name)
        .load_offsets(&args.source_name)
        .load_outdegrees()
        .build();

    if !args.perf_test {
        let comp_time = Instant::now();
        bvgraph.store(&args.dest_name.unwrap()).expect("Failed storing the graph");
        let comp_time = comp_time.elapsed().as_nanos() as f64;
        println!("decompressed the graph in {}ns", comp_time);
    } else {
        let mut rng = rand::thread_rng();
        let queries: Vec<_> = (0..N_QUERIES)
                                .map(|_| rng.gen_range(0..bvgraph.num_nodes()))
                                .collect();

        let mut ibs = BinaryReader::new(bvgraph.graph_memory.clone());

        let mut huff_decoder = HuffmanDecoder::new();
        huff_decoder.decode_headers(&mut ibs, NUM_CONTEXTS);

        let total = Instant::now();
        for &query in queries.iter() {
            bvgraph.decode_list(query, &mut ibs, None, &mut [], &mut huff_decoder);
        }
        let avg_query = (total.elapsed().as_nanos() as f64) / N_QUERIES as f64;
    
        println!("time per query: {}ns", avg_query);
    }
}