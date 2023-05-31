mod tests;

use std::{fs, str::FromStr, fmt, path::Iter, marker::PhantomData, iter::Map};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::ImmutableGraph;

#[derive(Serialize, Deserialize)]
pub struct UncompressedGraph<T> {
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
    fn outdegree(&self, x: Self::NodeT) -> Option<usize> {
        // offsets are correlated with the number of nodes since a node that has
        // no outgoing edges is represented in graph_memory anyway
        if x < T::zero() || x.to_usize().unwrap() > self.n {
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

impl<T> UncompressedGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    fn successors(&self, x: T) -> UncompressedGraphIterator<T, &UncompressedGraph<T>> {
        todo!()
    }
}

impl<T> AsRef<UncompressedGraph<T>> for UncompressedGraph<T> {
    fn as_ref(&self) -> &UncompressedGraph<T> {
        self
    }
}

impl<T> IntoIterator for UncompressedGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    type Item = T;

    type IntoIter = UncompressedGraphIterator<T, UncompressedGraph<T>>;

    fn into_iter(self) -> Self::IntoIter {
        UncompressedGraphIterator {
            curr_node: T::zero(),
            curr_successors_num: None,
            curr_successor_idx: 0,
            graph: self
        }
    }
}

pub struct UncompressedGraphIterator<T, UG: AsRef<UncompressedGraph<T>>> {
    curr_node: T,
    curr_successors_num: Option<usize>,
    curr_successor_idx: usize,
    graph: UG,
}

impl<T, UG: AsRef<UncompressedGraph<T>>> Iterator for UncompressedGraphIterator<T, UG>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive 
        + serde::Serialize
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let g = self.graph.as_ref();

        if self.curr_successors_num.is_none() {
            self.curr_successors_num = g.outdegree(self.curr_node);
        }
        
        let left = g.offsets[self.curr_node.to_usize().unwrap()];

        if left + self.curr_successor_idx > self.curr_successors_num.unwrap() {
            return None;
        } 

        self.curr_successor_idx += 1;
        Some(g.graph_memory[left + self.curr_successor_idx - 1])
    }
}

#[derive(Serialize, Deserialize)]
struct ImmutableGraphBuilder<T> {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<T>,
    loaded_offsets: Vec<usize>,
}

impl<T> ImmutableGraphBuilder<T>
where 
    T: num_traits::PrimInt + FromStr,
    <T as FromStr>::Err: fmt::Debug,
    T: DeserializeOwned
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
    fn load_graph(mut self, filename: &str) -> Self {
        self.loaded_graph = fs::read_to_string(format!("{}.graph.plain", filename))
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
    fn load_graph_bin(mut self, filename: &str) -> Self {
        let file = fs::read(format!("{}.graph.bin", filename)).expect("Failed reading the graph file");
        
        self.loaded_graph = bincode::deserialize(&file).expect("Error in deserializing the graph file");

        self
    }

    /// Loads the offsets file represented in plain mode (i.e. decimal numbers).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the offsets file
    fn load_offsets(mut self, filename: &str) -> Self {
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

    /// Loads the offsets file represented in binary mode (serialized).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the graph file
    fn load_offsets_bin(mut self, filename: &str) -> Self {
        let f = fs::read(format!("{}.offsets.bin", filename)).expect("Failed reading the offsets file");
        self.loaded_offsets = bincode::deserialize(&f).expect("Error in deserializing the offsets file");

        self
    }

    /// Computes the number of nodes by counting the offsets' file entries.
    /// 
    /// This is correct since all the nodes are represented in the graph file, even those
    /// not having any successor. Hence, their positions are written in the offsets file,
    /// and the amount of entries corresponds to the total amount of nodes. 
    fn count_nodes(mut self) -> Self {
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        self.num_nodes = self.loaded_offsets.len();
        self
    }

    fn count_arcs(mut self) -> Self {
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
