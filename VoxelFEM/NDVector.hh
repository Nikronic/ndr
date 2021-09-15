////////////////////////////////////////////////////////////////////////////////
// NDVector.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  ND Vector class supporting scanline-order traversal (calling a
//  visitor function for each entry).
//
//  Elements can be accessed in two ways:
//      list of index args:    array(i0, i1, ...)
//      vector of indices:     array(NDVectorIndex<N>)
//  Visitors applied to an array can have three signatures:
//      value-only:            f(val)
//      value and index list:  f(val, i0, i1, ...)
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef NDVector_HH
#define NDVector_HH
#include <vector>
#include <algorithm>
#include <MeshFEM/Types.hh>
#include <MeshFEM/Future.hh>
#include <MeshFEM/function_traits.hh>
#include <MeshFEM/TemplateHacks.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <iostream>
#include <cassert>

template<typename T>
class NDVector {
public:

    // Empty constructor -- leaves the data initialized
    NDVector() {}

    // Initializes only the dimensions of NDVector -- leaves the data initialized
    template<typename SizeContainer>
    NDVector(const SizeContainer &sizes) {
        Resize(sizes);
    }

    // Initializes only the dimensions of NDVector -- leaves the data initialized
    template<typename... Args>
    NDVector(Args... _sizes) : NDVector(std::vector<size_t>({size_t(_sizes)...})) {
        static_assert(all_integer_parameters<Args...>(), "NDVector constructor parameters must all be integers.");
    }

    // Set the dimensions of the NDVector
    // Computes the corresponding number of elements inside m_totalNumberOfElements
    // Resize the data std::vector accordingly
    void Resize(const std::vector<size_t> &sizes) {
        if (sizes.size() == 0) throw std::runtime_error("NDVector must have at least one dimension");
        if (std::find(sizes.begin(), sizes.end(), 0) != sizes.end())
            throw std::runtime_error("NDVector dimensions should be strictly positive");

        // Initialize array storing the Dimensions
        m_nb_dimension = sizes.size();
        m_nbElement_per_dimension = sizes;

        // Initialize the array storing the elements
        m_totalNumberOfElements = totalNumberOfElements();
        m_data.resize(m_totalNumberOfElements);
    }

    template<typename Derived>
    void Resize(const Eigen::PlainObjectBase<Derived> &sizes) {
        static_assert(std::is_same<typename Derived::Scalar, size_t>::value, "Elements of sizes must be of type size_t");
        static_assert((Derived::ColsAtCompileTime == 1) || (Derived::RowsAtCompileTime == 1), "sizes must be a vector.");
        Resize(std::vector<size_t>(sizes.data(), sizes.data() + sizes.size()));
    }

    // Return the total number of elements stored in the NDVector
    size_t Size() const { return m_totalNumberOfElements; }

    // Round brackets are for multi-index accessing
    template<typename... Args> const T &operator()(Args... Indices) const { return m_data[flatIndex(std::array<FirstType<Args...>, sizeof...(Indices)>{FirstType<Args...>(Indices)...})]; }
    template<typename... Args>       T &operator()(Args... Indices)       { return m_data[flatIndex(std::array<FirstType<Args...>, sizeof...(Indices)>{FirstType<Args...>(Indices)...})]; }
    template<typename indexType> const T &operator()(std::vector<indexType> Indices) const { return m_data[flatIndex(Indices)]; }
    template<typename indexType>       T &operator()(std::vector<indexType> Indices)       { return m_data[flatIndex(Indices)]; }

    // Square brackets are for linear index accessing
    const T &operator[](size_t idx1D) const { return m_data[idx1D]; }
          T &operator[](size_t idx1D)       { return m_data[idx1D]; }

    // Direct accessor to the data
    const aligned_std_vector<T> &data() const { return m_data; }
          aligned_std_vector<T> &data()       { return m_data; }

    Eigen::Map<      Eigen::VectorXd> flattened()       { return Eigen::Map<      Eigen::VectorXd>(m_data.data(), m_data.size()); }
    Eigen::Map<const Eigen::VectorXd> flattened() const { return Eigen::Map<const Eigen::VectorXd>(m_data.data(), m_data.size()); }

    // Visit each entry in scanline order, calling either visitor(val, idx0, idx1, ...)
    // or visitor(val) depending on visitor's signature
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 2, void>::type
    visit(F &&visitor) {
        for(size_t index = 0; index < m_totalNumberOfElements; ++index)
            visitor(m_data[index], unflattenIndex(index) );
    }
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 1, void>::type
    visit(F &&visitor) {
        for(size_t index = 0; index < m_totalNumberOfElements; ++index)
            visitor(m_data[index]);
    }
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 2, void>::type
    visit(F &&visitor) const {
        for(size_t index = 0; index < m_totalNumberOfElements; ++index)
            visitor(m_data[index], unflattenIndex(index) );

    }
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 1, void>::type
    visit(F &&visitor) const {
        for(size_t index = 0; index < m_totalNumberOfElements; ++index)
            visitor(m_data[index]);
    }


    void fill(const T &value) { for(auto & m_value : m_data) m_value = value; }
    template<typename DataContainer>
    void fill(const DataContainer &values) {
        if (values.size() != m_data.size())
            throw std::runtime_error("Can not fill NDvector of size " + std::to_string(m_data.size())
                                     + " with input vector of size " + std::to_string(values.size()));
        for (size_t i = 0; i < values.size(); ++i)
            m_data[i] = values[i];
    }

    void swap(NDVector<T> &other) {
        std::swap(other.m_nbElement_per_dimension, this->m_nbElement_per_dimension);
        std::swap(other.m_data, this->m_data);
        std::swap(other.m_nb_dimension, this->m_nb_dimension);
        std::swap(other.m_totalNumberOfElements, this->m_totalNumberOfElements);
    }

    // Compute, for each element of the array, the average difference of this element with its neighbors
    // Neighbors are the elements with a difference of 1 in one index of the multi-index notation
    // eg in 2D, element (i,j) neighbors are (i-1,j), (i+1,j), (i,j-1), (i,j+1)
    // \return NDVector of the all the average difference to neighbors
    NDVector<T> differenceToNeighborsAverage() const {
        NDVector<T> result(m_nbElement_per_dimension);

        for (size_t ei = 0 ; ei < m_totalNumberOfElements; ++ei) {
            std::vector<size_t> NDindices = unflattenIndex(ei);
            size_t nbNeighbors = 0;
            T average = 0;
            for (size_t dim = 0; dim < m_nb_dimension; ++dim) {
                if (NDindices[dim] == 0) {
                    // no "left" neighbour
                    NDindices[dim]++;
                    average += (*this)(NDindices);
                    NDindices[dim]--;
                    nbNeighbors++;
                } else if (NDindices[dim] == m_nbElement_per_dimension[dim]-1) {
                    // no "right" neighbour
                    NDindices[dim]--;
                    average += (*this)(NDindices);
                    NDindices[dim]++;
                    nbNeighbors++;
                } else {
                    // Both neighbours exist
                    NDindices[dim]--;
                    average += (*this)(NDindices);
                    NDindices[dim] += 2;
                    average += (*this)(NDindices);
                    NDindices[dim]--;
                    nbNeighbors += 2;
                }
            }
            average /= nbNeighbors;
            result[ei] = average - m_data[ei];
        }
        return result;
    }

    void visitLayer(size_t layerIndex, const std::function<void(size_t)> &callback) const {
        return visitLayer(layerIndex, m_nbElement_per_dimension, callback);
    }

    void visitSupportingRegion(std::vector<size_t> voxelIndex, const std::function<void(size_t)> &callback) const {
        return visitSupportingRegion(voxelIndex, m_nbElement_per_dimension, callback);
    }

    template<typename Callback>
    void visitNeighbors(size_t flatIndex, double radius, Callback &&callback) const {
        return visitNeighbors(flatIndex, m_nbElement_per_dimension, radius, callback);
    }

    template<typename SizeContainer>
    static void visitLayer(size_t layerIndex, const SizeContainer &sizes, const std::function<void(size_t)> &callback) {
        size_t N = sizes.size();
        if ((layerIndex < 0) || (layerIndex > sizes[N-1]-1))
            throw std::runtime_error("NDVector has no layer " + std::to_string(layerIndex) + ".");
        if (N == 2) {
            std::vector<size_t> idx = {0, layerIndex};
            for (size_t i = 0; i < sizes[0]; i++) {
                idx[0] = i;
                callback(NDVector<T>::flatIndex(idx, sizes));
            }
        }
        else if (N == 3) {
            std::vector<size_t> idx = {0, 0, layerIndex};
            for (size_t i = 0; i < sizes[0]; i++) {
                idx[0] = i;
                for (size_t j = 0; j < sizes[1]; j++) {
                    idx[1] = j;
                    callback(NDVector<T>::flatIndex(idx, sizes));
                }
            }          
        }
    }

    template<typename SizeContainer>
    static void visitSupportingRegion(std::vector<size_t> voxelIndex, const SizeContainer &sizes, const std::function<void(size_t)> &callback) {
        size_t N = sizes.size();
        if (voxelIndex[N-1] == 0)
            throw std::runtime_error("Lower layer has no supporting region.");
        std::vector<int> supportCenter(voxelIndex.cbegin(), voxelIndex.cend()); // cast to int: indices could become negative
        supportCenter[N-1] -= 1;
        std::vector<int> currentIndex(supportCenter);
        callback(NDVector<T>::flatIndex(currentIndex, sizes)); // voxel below
        for(size_t d = 0; d < N-1; d++) {                      // "side" voxels
            currentIndex = supportCenter;
            currentIndex[d] -= 1;
            if(NDIndexInBound(currentIndex, sizes))
                callback(NDVector<T>::flatIndex(currentIndex, sizes));
            currentIndex[d] += 2;
            if(NDIndexInBound(currentIndex, sizes))
                callback(NDVector<T>::flatIndex(currentIndex, sizes));
        }
    }

    template<typename SizeContainer, typename Callback>
    static void visitNeighbors(size_t flatIndex, const SizeContainer &sizes, double radius, Callback &&callback) {
        std::vector<size_t> centerIndex = NDVector<T>::unflattenIndex(flatIndex, sizes);
        size_t N = sizes.size();
        size_t neighborhoodSize = std::pow(2*radius+1, N);
        std::vector<int> neighborIndex(N); // type is int because indices could become negative (neighborhood gets outside the domain)
        std::vector<int> loBounds(N);
        std::vector<int> upBounds(N);
        for(size_t d = 0; d < N; d++) {
            loBounds[d] = centerIndex[d] - radius;
            upBounds[d] = centerIndex[d] + radius;
        }
        neighborIndex = loBounds;
        for(size_t count = 0; count < neighborhoodSize; count++) {
            if(NDIndexInBound(neighborIndex, sizes))
                callback(NDVector<T>::flatIndex(neighborIndex, sizes));
            updateNDigitCounter(neighborIndex, loBounds, upBounds);
        }
    }

    static void updateNDigitCounter(std::vector<int> &counter, const std::vector<int> &loBounds, const std::vector<int> &upBounds) {
        size_t N = counter.size();
        for (size_t d = 0; d < N; ++d) {
            ++counter[d];
            if (counter[d] <= upBounds[d]) break;
            counter[d] = loBounds[d];
        }
    }

    // Check if a grid (unflattened) index identifies an element of the NDVector or is out of bound (in a NDVector of specified size)
    template<typename IndexContainer, typename SizeContainer>
    static bool NDIndexInBound(const IndexContainer &gridIndices, const SizeContainer &sizes) {
        if (gridIndices.size() != sizes.size())
            throw std::runtime_error("Invalid number of indices, got " + std::to_string(sizes.size())
                                     + " indices but got " + std::to_string(gridIndices.size()) + " sizes");
        for(size_t d = 0; d < sizes.size(); d++)
            if((gridIndices[d] < 0) || (gridIndices[d] >= sizes[d]))
                return false;
        return true;
    }

    // Get the linear index for an NDVector with dimensions `sizes`
    // corresponding to the input multi-indices stored in `indices`.
    // Template parameters IndexContainer and SizeContainer are, e.g., std::vector<size_t> or std::array<size_t, N>.
    template<typename IndexContainer, typename SizeContainer>
    static size_t flatIndex(const IndexContainer &indices, const SizeContainer &sizes) {
        if (!(NDIndexInBound(indices, sizes)))
            throw std::runtime_error("Indices out of bounds of NDVector");

        // (i)     -> i
        // (i,j)   -> Ny * i + j
        // (i,j,k) -> Nz * (Ny * i + j) + k
        const size_t dims = indices.size();
        size_t result = 0;
        for (size_t d = 0; d < dims; ++d)
            result = result * sizes[d] + indices[d];

        return result;
    }

    // Get the linear index corresponding to the input multi-indices stored in a std::vector (in a NDVector of the same
    // dimensions as *this)
    template<typename IndexContainer>
    size_t flatIndex(const IndexContainer &indices) const {
        size_t result = flatIndex(indices, m_nbElement_per_dimension);
        if (result >= m_totalNumberOfElements)
            throw std::runtime_error("Index out of range");
        return result;
    }

    // Get the multi-index of the data corresponding to the input linear index (in a NDVector of the same dimensions as *this)
    std::vector<size_t> unflattenIndex (size_t flatIndex) const {
        return unflattenIndex(flatIndex, m_nbElement_per_dimension);
    }

    template<size_t N>
    Eigen::Array<size_t, N, 1> unflattenIndex(size_t flatIndex) const {
        Eigen::Array<size_t, N, 1> result;
        unflattenIndex(flatIndex, m_nbElement_per_dimension, result);
        return result;
    }

    template<typename SizeContainer, typename ResultContainer>
    static void unflattenIndex(size_t flatIndex, const SizeContainer &sizes, ResultContainer &result) {
        const size_t dims = sizes.size();
        result.resize(dims);
        for (size_t d = 0; d < dims; ++d) {
            size_t s = sizes[(dims - 1) - d];
            result[(dims - 1) - d] = flatIndex % s;
            flatIndex /= s;
        }

        if (flatIndex != 0)
            throw std::runtime_error("Index out of range");

    }

    // Get the multi-index of the data corresponding to the input linear index, in a NDVector of dimensions given by
    // input "sizes"
    // SizeContainer is, e.g., std::vector<size_t>
    template<typename SizeContainer>
    static std::vector<size_t> unflattenIndex(size_t flatIndex, const SizeContainer &sizes) {
        std::vector<size_t> result;
        unflattenIndex(flatIndex, sizes, result);
        return result;
    }

    // Returns: the number of elements in each dimension
    const std::vector<size_t> &NbComponentPerDimensions() const { return m_nbElement_per_dimension; }
    // Increment in flat index induced by changing an ND-index
    template<class IndexContainer>
    const void getFlatIndexIncrements(IndexContainer &result) const {
        const size_t N = m_nb_dimension;
        result.resize(N);
        result[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            result[d] = result[d + 1] * m_nbElement_per_dimension[d + 1];
    }

    // Iterators
    typename aligned_std_vector<T>::iterator begin() { return m_data.begin(); }
    typename aligned_std_vector<T>::iterator   end() { return m_data.end(); }
    typename aligned_std_vector<T>::const_iterator begin() const { return m_data.begin(); }
    typename aligned_std_vector<T>::const_iterator   end() const { return m_data.end(); }


private:

    // Get the linear index corresponding to the input multi-indices (in a NDVector of the same dimensions as *this)
    template<typename... Args> size_t flatIndex(Args... Indices) const {
        if (sizeof...(Indices) != m_nb_dimension)
            throw std::runtime_error("Invalid number of indices, expected "
                                     + std::to_string(m_nb_dimension)
                                     + " but got " + std::to_string(sizeof...(Indices)));
        std::vector<size_t> indices{{size_t(Indices)...}};
        return flatIndex(indices);
    }

    // Return the total number of elements that can be stored in m_data
    size_t totalNumberOfElements() const {
        size_t nbElements = 1;
        for (auto const & dim : m_nbElement_per_dimension) {
            nbElements *= dim;
        }
        return nbElements;
    }

    // member to hold the size of m_data
    size_t m_totalNumberOfElements;

    // number of dimensions: "N" of NDVector
    size_t m_nb_dimension;
    // number of elements to store in each dimension
    std::vector<size_t> m_nbElement_per_dimension;

    // The actual storage member
    aligned_std_vector<T> m_data;
};


#endif /* end of include guard: NDVector_HH */
