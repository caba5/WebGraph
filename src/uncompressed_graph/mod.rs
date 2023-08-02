use std::{fs, str::FromStr, fmt::{self}, marker::PhantomData};

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::ImmutableGraph;

#[derive(Serialize, Deserialize, Clone)]
pub struct UncompressedGraph<T> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    n: usize,
    m: usize,
    pub graph_memory: Vec<T>,
    pub offsets: Vec<usize>,
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
    /// Since the nodes are numbered from *0* to *n - 1*, asking for the outdegree of node *x* 
    /// corresponds to asking the outdegree of the *x*-th node (zero-indexed).
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number (from *0* to *n - 1*)
    /// 
    /// # Examples
    /// 
    /// ```
    /// let uncompressed_graph = build_graph_bin::<u32>();
    /// uncompressed_graph.outdegree(5);
    /// ```
    fn outdegree(&mut self, x: Self::NodeT) -> Option<usize> {
        // Offsets also represent nodes having no out-edges.
        // The argument node has to be lower than the last offsets position since the latter
        // is left just for convenience and does not represent any real position in the graph.
        if x < T::zero() || x.to_usize().unwrap() > self.offsets.len() - 1 {
            return None;
        }
        
        let usized_ind = x.to_usize().unwrap();
        let left_index = if usized_ind == 0 {0} else {self.offsets[usized_ind - 1]};

        if left_index < self.graph_memory.len() {
            let right_index = self.offsets[if usized_ind == 0 {0} else {usized_ind}];
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
    /// 
    /// # Examples
    /// 
    /// ```
    /// let uncompressed_graph = build_graph_bin::<u32>();
    /// let result = uncompressed_graph.store("graph_name");
    /// ```
    fn store(&mut self, filename: &str) -> std::io::Result<()>{
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
        + Clone
{
    /// Returns an `UncompressedGraphIterator` over the successors of `x`.
    /// 
    /// # Arguments
    /// 
    /// * `x` - The node number (from *0* to *n - 1*)
    /// 
    /// # Examples
    /// 
    /// ```
    /// let uncompressed_graph = build_graph_bin::<u32>();
    /// let successors_it = uncompressed_graph.successors(5);
    /// 
    /// while let Some(node_val) = successors_it.next() {
    ///     println!("Successor: {}", node_val);
    /// }
    /// ```
    pub fn successors(&mut self, x: T) -> UncompressedGraphIterator<T, &UncompressedGraph<T>> {
        let base = if x == T::zero() {0} else {self.offsets[x.to_usize().unwrap() - 1]};
        UncompressedGraphIterator { 
            base,
            idx_from_base: 0,
            up_to: base + self.outdegree(x).unwrap() - 1,
            graph: self,
            _phantom: PhantomData
        }
    }

    pub fn outdegree_internal(&self, x: T) -> usize {
        let usized_ind = x.to_usize().unwrap();
        let left_index = if usized_ind == 0 {0} else {self.offsets[usized_ind - 1]};

        if left_index < self.graph_memory.len() {
            let right_index = self.offsets[if usized_ind == 0 {0} else {usized_ind}];
            return right_index - left_index - 1;
        }
         
        self.graph_memory.len() - left_index - 1
    }
}

impl<T> AsRef<UncompressedGraph<T>> for UncompressedGraph<T>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
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
        + Clone
{
    type Item = T;

    type IntoIter = UncompressedGraphIterator<T, UncompressedGraph<T>>;

    fn into_iter(self) -> Self::IntoIter {
        UncompressedGraphIterator {
            base: 0,
            idx_from_base: 0,
            up_to: self.graph_memory.len(),
            graph: self,
            _phantom: PhantomData
        }
    }
}

pub struct UncompressedGraphIterator<T, UG: AsRef<UncompressedGraph<T>>> 
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive
        + serde::Serialize
{
    base: usize,
    idx_from_base: usize,
    up_to: usize,
    graph: UG,
    _phantom: PhantomData<T>
}

impl<T, UG: AsRef<UncompressedGraph<T>>> Iterator for UncompressedGraphIterator<T, UG>
where T: 
        num_traits::Num 
        + PartialOrd 
        + num_traits::ToPrimitive 
        + serde::Serialize
        + Clone
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.base + self.idx_from_base > self.up_to {
            return None;
        }

        let g = self.graph.as_ref();
        
        self.idx_from_base += 1;        
        
        Some(g.graph_memory[self.base + self.idx_from_base].clone())
    }
}

#[derive(Serialize, Deserialize)]
pub struct ImmutableGraphBuilder<T> {
    num_nodes: usize,
    num_edges: usize,
    loaded_graph: Vec<T>,
    loaded_offsets: Vec<usize>,
}

impl<T> Default for ImmutableGraphBuilder<T>
where
    T: num_traits::PrimInt + FromStr,
    <T as FromStr>::Err: fmt::Debug,
    T: DeserializeOwned ,
    T: Serialize
{
    fn default() -> Self {
        Self { 
            num_nodes: 0, 
            num_edges: 0, 
            loaded_graph: Vec::default(), 
            loaded_offsets: Vec::default() 
        }
    }
}

impl<T> ImmutableGraphBuilder<T>
where 
    T: num_traits::PrimInt + FromStr,
    <T as FromStr>::Err: fmt::Debug,
    T: Serialize,
    T: DeserializeOwned
{
    pub fn new() -> ImmutableGraphBuilder<T> {
        Self::default()
    }

    /// Loads a graph file represented in plain mode (i.e. decimal numbers).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the graph file
    pub fn load_graph(mut self, filename: &str) -> Self {
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
    pub fn load_graph_bin(mut self, filename: &str) -> Self {
        let file = fs::read(format!("{}.graph.bin", filename)).expect("Failed reading the graph file");
        
        self.loaded_graph = bincode::deserialize(&file).expect("Error in deserializing the graph file");

        self
    }

    /// Loads the offsets file represented in plain mode (i.e. decimal numbers).
    /// 
    /// # Arguments
    /// 
    /// * `filename` - The filename of the offsets file
    pub fn load_offsets(mut self, filename: &str) -> Self {
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
    pub fn load_offsets_bin(mut self, filename: &str) -> Self {
        let f = fs::read(format!("{}.offsets.bin", filename)).expect("Failed reading the offsets file");
        self.loaded_offsets = bincode::deserialize(&f).expect("Error in deserializing the offsets file");

        self
    }

    /// Computes the number of nodes by counting the offsets' file entries.
    /// 
    /// This is correct since all the nodes are represented in the graph file, even those
    /// not having any successor. Hence, their positions are written in the offsets file,
    /// and the amount of entries corresponds to the total amount of nodes. 
    pub fn count_nodes(mut self) -> Self {
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        self.num_nodes = self.loaded_offsets.len();
        self
    }

    /// Computes the number of edges of the graph.
    pub fn count_arcs(mut self) -> Self {
        assert!(!self.loaded_graph.is_empty(), "The graph has to be loaded");
        assert!(!self.loaded_offsets.is_empty(), "The offsets have to be loaded");

        self.num_edges = self.loaded_graph.len() - self.loaded_offsets.len();

        self
    }

    /// Constructs the UncompressedGraph object.
    pub fn build(self) -> UncompressedGraph<T> {
        UncompressedGraph::<T> { 
            n: self.num_nodes, 
            m: self.num_edges, 
            graph_memory: self.loaded_graph, 
            offsets: self.loaded_offsets 
        }
    }
}

#[cfg(test)]
mod tests;