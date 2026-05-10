use super::header::IndexHeader;
use super::stored_metadata::StoredChunkMetadata;

#[derive(Debug, Clone, PartialEq)]
pub struct VectorStore {
    pub(crate) data: Vec<f32>,
    pub(crate) dims: usize,
    pub(crate) count: usize,
}

impl VectorStore {
    pub fn from_vec_vec(vecs: Vec<Vec<f32>>) -> anyhow::Result<Self> {
        let count = vecs.len();
        if count == 0 {
            return Ok(Self { data: vec![], dims: 0, count: 0 });
        }
        let dims = vecs[0].len();
        let mut data = Vec::with_capacity(count * dims);
        for v in vecs {
            anyhow::ensure!(v.len() == dims, "inconsistent vector dimensions");
            data.extend_from_slice(&v);
        }
        Ok(Self { data, dims, count })
    }

    pub fn get(&self, i: usize) -> &[f32] {
        let start = i * self.dims;
        &self.data[start..start + self.dims]
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn into_vec_vec(self) -> Vec<Vec<f32>> {
        let VectorStore { data, dims, count } = self;
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * dims;
            result.push(data[start..start + dims].to_vec());
        }
        result
    }

    pub fn as_bytes(&self) -> &[u8] {
        if self.data.is_empty() {
            return &[];
        }
        bytemuck::cast_slice(&self.data)
    }

    pub fn concat(a: &VectorStore, b: &VectorStore) -> anyhow::Result<Self> {
        anyhow::ensure!(
            a.dims == b.dims || a.is_empty() || b.is_empty(),
            "dimension mismatch: {} vs {}",
            a.dims(),
            b.dims()
        );
        let dims = if a.is_empty() { b.dims() } else { a.dims() };
        let mut data = Vec::with_capacity((a.count + b.count) * dims);
        data.extend_from_slice(&a.data);
        data.extend_from_slice(&b.data);
        Ok(Self { data, dims, count: a.count + b.count })
    }
}

#[derive(Debug)]
pub struct StoredIndex {
    pub header: IndexHeader,
    pub vectors: VectorStore,
    pub metadata: Vec<StoredChunkMetadata>,
}
