use webgraph_rust::ascii_graph::{AsciiGraphBuilder, AsciiGraph};
use webgraph_rust::properties::Properties;
use webgraph_rust::utils::encodings::{UniversalCode, GammaCode, UnaryCode, ZetaCode, DeltaCode};
use webgraph_rust::webgraph::bvgraph::{BVGraph, BVGraphBuilder};
use webgraph_rust::{EncodingType, ImmutableGraph};

use core::panic;
use std::fs::{self, File};
use std::io::BufReader;
use std::time::Instant;

use clap::Parser;

use rand::Rng;

#[derive(Parser, Debug)]
struct WGArgs { // TODO: implement reading in one coding and writing into a different coding 
    /// The size of the window
    #[arg(short = 'w', long = "window-size", default_value_t = 7)]
    window_size: usize,
    /// The maximum reference chain length
    #[arg(short = 'r', long = "max-ref-count", default_value_t = 3)]
    max_ref_count: usize,
    /// The minimum length of the interval
    #[arg(short = 'i', long = "min-interval-len", default_value_t = 4)]
    min_interval_length: usize,
    /// Specifies the block coding type
    #[arg(long = "block-coding", default_value_t = EncodingType::GAMMA)]
    block_coding: EncodingType,
    /// Specifies the block count coding type
    #[arg(long = "block-count-coding", default_value_t = EncodingType::GAMMA)]
    block_count_coding: EncodingType,
    /// Specifies the outdegree coding type
    #[arg(long = "outdegree-coding", default_value_t = EncodingType::GAMMA)]
    outdegree_coding: EncodingType,
    /// Specifies the offset coding type
    #[arg(long = "offset-coding", default_value_t = EncodingType::GAMMA)]
    offset_coding: EncodingType,
    /// Specifies the reference coding type
    #[arg(long = "reference-coding", default_value_t = EncodingType::UNARY)]
    reference_coding: EncodingType,
    /// Specifies the intervals coding type
    #[arg(long = "interval-coding", default_value_t = EncodingType::GAMMA)]
    interval_coding: EncodingType,
    /// Specifies the residual coding type
    #[arg(long = "residual-coding", default_value_t = EncodingType::ZETA)]
    residual_coding: EncodingType,
    /// Specifies the k parameter for ZetaK coding
    #[arg(short = 'k', long = "zetak", default_value_t = 3)]
    zeta_k: usize,
    /// Whether to compress offsets through Elias-Fano. It overwrites the offset_coding, if specified.
    #[arg(long = "ef", default_value_t = false)]
    elias_fano: bool,
    /// Source basename
    source_name: String,                                                    // TODO: PathBuf instead of str?
    /// Destination basename
    dest_name: Option<String>,                                              // TODO: PathBuf instead of str?
    /// Check that the written graph is equal to the read graph
    #[arg(short, long = "check", default_value_t = false)]
    check: bool,
    /// Performance test
    #[arg(short, long = "perf", default_value_t = false)]
    perf_test: bool,
    /// Compress starting from an ASCII graph
    #[arg(short = 'a', long = "ascii", default_value_t = false)]
    from_plain: bool,
}

const N_QUERIES: usize = 1000000;

fn gen_queries(n_queries: usize, range_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    (0..n_queries)
        .map(|_| rng.gen_range(0..range_size))
        .collect()
}

fn decompression_perf_test<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutDegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutDegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
>(bvgraph: &mut BVGraph<
    InBlockCoding,
    InBlockCountCoding,
    InOutDegreeCoding,
    InOffsetCoding,
    InReferenceCoding,
    InIntervalCoding,
    InResidualCoding,
    OutBlockCoding,
    OutBlockCountCoding,
    OutOutDegreeCoding,
    OutOffsetCoding,
    OutReferenceCoding,
    OutIntervalCoding,
    OutResidualCoding,
>) {
    let n = bvgraph.num_nodes();

    let queries = gen_queries(N_QUERIES, n - 1);

    let total = Instant::now();
    for &query in queries.iter() {
        bvgraph.decode_list(query, &mut bvgraph.graph_binary_wrapper.borrow_mut(), None, &mut []);
    }
    let avg_query = (total.elapsed().as_nanos() as f64) / N_QUERIES as f64;

    println!("time per query: {}ns", avg_query);
}

fn create_graph<
    InBlockCoding: UniversalCode,
    InBlockCountCoding: UniversalCode,
    InOutDegreeCoding: UniversalCode,
    InOffsetCoding: UniversalCode,
    InReferenceCoding: UniversalCode,
    InIntervalCoding: UniversalCode,
    InResidualCoding: UniversalCode,
    OutBlockCoding: UniversalCode,
    OutBlockCountCoding: UniversalCode,
    OutOutDegreeCoding: UniversalCode,
    OutOffsetCoding: UniversalCode,
    OutReferenceCoding: UniversalCode,
    OutIntervalCoding: UniversalCode,
    OutResidualCoding: UniversalCode,
>(props: &Properties, in_name: &str, out_name: Option<String>, elias_fano: bool, perf_test: bool, check: bool, plain_graph: Option<AsciiGraph<usize>>) {
    if let Some(plain_graph) = plain_graph {
        let bvgraph = BVGraphBuilder::<
            InBlockCoding,
            InBlockCountCoding,
            InOutDegreeCoding,
            InOffsetCoding,
            InReferenceCoding,
            InIntervalCoding,
            InResidualCoding,
            OutBlockCoding,
            OutBlockCountCoding,
            OutOutDegreeCoding,
            OutOffsetCoding,
            OutReferenceCoding,
            OutIntervalCoding,
            OutResidualCoding,
        >::new()
            .set_min_interval_len(props.min_interval_len)
            .set_max_ref_count(props.max_ref_count)
            .set_window_size(props.window_size)
            .set_zeta(props.zeta_k)
            .set_num_nodes(plain_graph.num_nodes())
            .set_num_edges(plain_graph.num_arcs())
            .build();

        let comp_time = Instant::now();
        bvgraph.store_plain(&plain_graph, out_name.unwrap().as_str()).expect("Failed storing the plain graph");
        let comp_time = comp_time.elapsed().as_nanos() as f64;
        println!("compressed the plain graph in {}ns", comp_time);
    } else {
        let mut bvgraph = BVGraphBuilder::<
            InBlockCoding,
            InBlockCountCoding,
            InOutDegreeCoding,
            InOffsetCoding,
            InReferenceCoding,
            InIntervalCoding,
            InResidualCoding,
            OutBlockCoding,
            OutBlockCountCoding,
            OutOutDegreeCoding,
            OutOffsetCoding,
            OutReferenceCoding,
            OutIntervalCoding,
            OutResidualCoding,
        >::new()
            .set_min_interval_len(props.min_interval_len)
            .set_max_ref_count(props.max_ref_count)
            .set_window_size(props.window_size)
            .set_zeta(props.zeta_k)
            .set_elias_fano(elias_fano)
            .set_num_nodes(props.nodes)
            .set_num_edges(props.arcs)
            .load_graph(in_name)
            .load_offsets(in_name)
            .load_outdegrees()
            .build();

        if perf_test {
            decompression_perf_test(&mut bvgraph);
        } else if let Some(out_name) = out_name{
            let comp_time = Instant::now();
            bvgraph.store(out_name.as_str()).expect("Failed storing the graph");
            let comp_time = comp_time.elapsed().as_nanos() as f64;
            println!("compressed the graph in {}ns", comp_time);

            if check {
                let compressed_graph = BVGraphBuilder::<
                    InBlockCoding,
                    InBlockCountCoding,
                    InOutDegreeCoding,
                    InOffsetCoding,
                    InReferenceCoding,
                    InIntervalCoding,
                    InResidualCoding,
                    OutBlockCoding,
                    OutBlockCountCoding,
                    OutOutDegreeCoding,
                    OutOffsetCoding,
                    OutReferenceCoding,
                    OutIntervalCoding,
                    OutResidualCoding,
                >::new()
                    .set_min_interval_len(props.min_interval_len)
                    .set_max_ref_count(props.max_ref_count)
                    .set_window_size(props.window_size)
                    .set_zeta(props.zeta_k)
                    .set_num_nodes(props.nodes)
                    .set_num_edges(props.arcs)
                    .load_graph(out_name.as_str())
                    .load_offsets(out_name.as_str())
                    .load_outdegrees()
                    .build();
                
                assert_eq!(bvgraph.graph_memory, compressed_graph.graph_memory);
                assert_eq!(bvgraph.offsets, compressed_graph.offsets);

                println!("Check passed");
            }
        } else {
            panic!("Neither an outname nor a performance test flag were provided");
        }
    }
}

fn main() {
    let mut args = WGArgs::parse();

    if args.perf_test && args.check {
        panic!("Both performance test and compression test flags were provided");
    }

    if args.check && args.dest_name.is_none() {
        panic!("Cannot execute a correctness check without compressing to file");
    }

    if args.perf_test && args.dest_name.is_some() {
        println!("Warning: both destination name and performance test flag provided, nothing will be written");
    }

    if args.from_plain && args.dest_name.is_none() {
        panic!("No destination name provided.");
    }

    let mut plain_graph = None;
    let mut props = Properties::default();
    
    if !args.from_plain {
        let properties_file = File::open(format!("{}.properties", args.source_name));
        let properties_file = properties_file.unwrap_or_else(|_| panic!("Could not find {}.properties", args.source_name));
        let p = java_properties::read(BufReader::new(properties_file)).unwrap_or_else(|_| panic!("Failed parsing the properties file"));

        props = Properties::from(p);

        props.window_size = args.window_size;
        props.max_ref_count = args.max_ref_count;
        props.min_interval_len = args.min_interval_length;
    } else {
        props.block_coding = args.block_coding;
        props.block_count_coding = args.block_count_coding;
        props.outdegree_coding = args.outdegree_coding;
        props.offset_coding = args.offset_coding;
        props.reference_coding = args.reference_coding;
        props.interval_coding = args.interval_coding;
        props.residual_coding = args.residual_coding;

        props.window_size = args.window_size;
        props.max_ref_count = args.max_ref_count;
        props.min_interval_len = args.min_interval_length;

        plain_graph = Some(AsciiGraphBuilder::new()
                        .load_ascii(&args.source_name)
                        .build());
    }

    create_graph::<GammaCode, GammaCode, GammaCode, GammaCode, UnaryCode, GammaCode, ZetaCode, GammaCode, GammaCode, ZetaCode, GammaCode, UnaryCode, ZetaCode, ZetaCode>
        (&props, &args.source_name, args.dest_name, args.elias_fano, args.perf_test, args.check, plain_graph); // Best compression
}