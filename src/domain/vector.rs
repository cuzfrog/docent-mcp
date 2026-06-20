/// A dense vector store — a flat `Vec<f32>` array with known dimension count.
///
/// The vectors are stored contiguously in row-major order:
/// `data[i * dims .. (i+1) * dims]` is the i-th vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub(crate) data: Vec<f32>,
    pub(crate) dims: usize,
    pub(crate) count: usize,
}

impl Vector {
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
        let Vector { data, dims, count } = self;
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

    pub fn concat(a: &Vector, b: &Vector) -> anyhow::Result<Self> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_vec_vec_roundtrip() {
        let orig = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let store = Vector::from_vec_vec(orig.clone()).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.dims(), 3);
        assert!(!store.is_empty());
        assert_eq!(store.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(store.get(1), &[4.0, 5.0, 6.0]);
        assert_eq!(store.get(2), &[7.0, 8.0, 9.0]);
        assert_eq!(store.into_vec_vec(), orig);
    }

    #[test]
    fn test_from_vec_vec_empty() {
        let store = Vector::from_vec_vec(vec![]).unwrap();
        assert_eq!(store.len(), 0);
        assert_eq!(store.dims(), 0);
        assert!(store.is_empty());
        assert_eq!(store.as_bytes(), &[] as &[u8]);
    }

    #[test]
    fn test_from_vec_vec_inconsistent_dims() {
        let result = Vector::from_vec_vec(vec![
            vec![1.0, 2.0],
            vec![3.0],
        ]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("inconsistent"));
    }

    #[test]
    fn test_concat_two_nonempty() {
        let a = Vector::from_vec_vec(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]).unwrap();
        let b = Vector::from_vec_vec(vec![
            vec![2.0, 0.0],
        ]).unwrap();
        let c = Vector::concat(&a, &b).unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c.dims(), 2);
        assert_eq!(c.get(0), &[1.0, 0.0]);
        assert_eq!(c.get(2), &[2.0, 0.0]);
    }

    #[test]
    fn test_concat_empty_left() {
        let a = Vector::from_vec_vec(vec![]).unwrap();
        let b = Vector::from_vec_vec(vec![vec![1.0]]).unwrap();
        let c = Vector::concat(&a, &b).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c.dims(), 1);
    }

    #[test]
    fn test_concat_empty_right() {
        let a = Vector::from_vec_vec(vec![vec![1.0]]).unwrap();
        let b = Vector::from_vec_vec(vec![]).unwrap();
        let c = Vector::concat(&a, &b).unwrap();
        assert_eq!(c.len(), 1);
        assert_eq!(c.dims(), 1);
    }

    #[test]
    fn test_concat_dimension_mismatch() {
        let a = Vector::from_vec_vec(vec![vec![1.0, 0.0]]).unwrap();
        let b = Vector::from_vec_vec(vec![vec![1.0]]).unwrap();
        let result = Vector::concat(&a, &b);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("dimension mismatch"));
    }

    #[test]
    fn test_as_bytes() {
        let store = Vector::from_vec_vec(vec![
            vec![1.0f32, 2.0f32],
        ]).unwrap();
        let bytes = store.as_bytes();
        assert_eq!(bytes.len(), 8); // 2 f32 * 4 bytes
    }

    #[test]
    fn test_get_out_of_bounds_panics() {
        let store = Vector::from_vec_vec(vec![vec![1.0, 2.0]]).unwrap();
        let result = std::panic::catch_unwind(|| store.get(5));
        assert!(result.is_err());
    }
}
