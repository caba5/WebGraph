mod tests;

use std::{fs, str::FromStr, fmt};

use serde::{Serialize, Deserialize};

use crate::ImmutableGraph;

#[derive(Serialize, Deserialize)]
struct UncompressedGraph<T> {
    n: usize,
    m: usize,
    graph_memory: Vec<T>,
    offsets: Vec<usize>,
}

impl<T> ImmutableGraph for UncompressedGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    type NodeT = T;

    fn num_nodes(&self) -> usize {
        self.n
    }

    fn num_arcs(&self) -> usize {
        self.m
    }

    /// Returns the number of adjacent nodes of the given node (if in bounds).
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number (from 0 to n)
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize> {
        // offsets are not correlated with the number of nodes since a node
        // can have no outgoing edges, thus it is not represented in graph_memory   // TODO: not true
        if x < T::zero() || x.to_usize().unwrap() > self.offsets.len() {
            return None;
        }
        
        let usized_ind = x.to_usize().unwrap();
        let left_index = self.offsets[usized_ind];

        if left_index < self.offsets.len() - 1 {
            let right_index = self.offsets[usized_ind + 1];
            return Some(right_index - left_index - 1);
        } 

        // If it is the last node having any successor, then
        // return the remainder of the graph_memory nodes list        
        Some(self.graph_memory.len() - left_index - 1)
    }

    fn successors(&self, x: Self::NodeT) -> Result<Box<dyn Iterator<Item = &u32>>, &str> {
        todo!()
    }

    /// Stores both `graph_memory` and `offsets` into their respective files.
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The name (with or without path) the saved files will have 
    fn store(&self, filename: &str) -> std::io::Result<()>{
        assert_ne!(filename, "");

        fs::write(format!("{}.graph.bin", filename), bincode::serialize(&self.graph_memory).unwrap())?;
        fs::write(format!("{}.offsets.bin", filename), bincode::serialize(&self.offsets).unwrap())?;

        Ok(())
    }

    // TODO: should the load be included?

}

#[derive(Serialize, Deserialize)]
struct ImmutableGraphBuilder<T> {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<T>,
    loaded_offsets: Vec<usize>,
}

impl<'a, T> ImmutableGraphBuilder<T>
where 
    T: num_traits::PrimInt + FromStr,
    <T as FromStr>::Err: fmt::Debug,
    T: serde::Deserialize<'a>
{
    fn new() -> ImmutableGraphBuilder<T> {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Vec::new(), 
            loaded_offsets: Vec::new() 
        }
    }

    /// Loads a graph file represented in plain mode (i.e. decimal numbers).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the graph file
    fn load_graph(mut self, filename: &str) -> ImmutableGraphBuilder<T>{
        self.loaded_graph = fs::read_to_string(filename)
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

    /// Loads a graph file represented in binary mode (serialized).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the graph file
    fn load_graph_bin(mut self, filename: &str) -> ImmutableGraphBuilder<T>{
        let f = fs::read(format!("{}.graph.bin", filename)).expect("Could not read the graph file");
        self.loaded_graph = bincode::deserialize::<Vec<T>>(f.as_slice()).expect("Could not deserialize the file");

        self
    }

    /// Loads the offsets file represented in plain mode (i.e. decimal numbers).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the offsets file
    fn load_offsets(mut self, filename: &str) -> ImmutableGraphBuilder<T> {
        self.loaded_offsets = fs::read_to_string(filename)
                            .expect("Failed to load the offsets file")
                            .split(' ')
                            .map(|node| node
                                                .parse()
                                                .unwrap_or_else(|_| panic!("Failed to parse offset {}", node))
                            )
                            .collect();
        self
    }

    /// Loads the offsets file represented in binary mode (serialized).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the graph file
    fn load_offsets_bin(mut self, filename: &str) -> ImmutableGraphBuilder<T>{
        let f = fs::read(format!("{}.offsets.bin", filename)).expect("Could not read the offsets file");
        self.loaded_offsets = bincode::deserialize::<Vec<usize>>(f.as_slice()).expect("Could not deserialize the file");

        self
    }

    /// Computes the number of nodes by counting the offsets' file entries.
    /// 
    /// This is correct since all the nodes are represented in the graph file, even those
    /// not having any successor. Hence, their positions are written in the offsets file,
    /// and the amount of entries corresponds to the total amount of nodes. 
    fn count_nodes(mut self) -> ImmutableGraphBuilder<T> {
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        self.num_nodes = self.loaded_offsets.len();
        self
    }

    fn count_arcs(mut self) -> ImmutableGraphBuilder<T> {
        assert!(!self.loaded_graph.is_empty(), "The graph has to be loaded");
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        self.num_edges = self.loaded_graph.len() - self.loaded_offsets.len();

        self
    }

    /// Constructs the UncompressedGraph object.
    fn construct(self) -> UncompressedGraph<T> {
        UncompressedGraph::<T> { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph, 
            offsets: self.loaded_offsets 
        }
    }
}
