#ifndef __RESERVOIR_WRAPPER_HPP__
#define __RESERVOIR_WRAPPER_HPP__


// wraps a model with an associated reservoir which stores a sub-sample
// of the symbols a model observes
template <typename Model, size_t ReservoirSize>
class ReservoirWrapper : public Compressor {

    public:

        constexpr static size_t ReservoirSlots = ReservoirSize;

        ReservoirWrapper();

        // the probability of seeing a particular symbol next
        virtual double prob(bit_t b);

        // the logarithm of the probability of all processed experience
        virtual double logBlockProbability() const;

        // process a new symbol
        virtual void update(bit_t b);

        // file extension
        virtual const char *fileExtension() const;

        // makes a copy of the model
        virtual Compressor *clone() const;

        // wrapped reservoir accessor
        const ReservoirSampling<bit_t> &reservoir() const;

    private:

        Model m_model;
        ReservoirSampling<bit_t> m_reservoir;
};



template <typename Model, size_t ReservoirSize>
inline ReservoirWrapper<Model, ReservoirSize>::ReservoirWrapper() :
    m_reservoir(ReservoirSize)
{
}


/* The probability of seeing a particular symbol next. */
template <typename Model, size_t ReservoirSize>
inline double ReservoirWrapper<Model, ReservoirSize>::prob(bit_t b) {

    return m_model.prob(b);
}


/* The logarithm of the probability of all seen symbols under the process. */
template <typename Model, size_t ReservoirSize>
inline double ReservoirWrapper<Model, ReservoirSize>::logBlockProbability() const {

    return m_model.logBlockProbability();
}


/* Process a new symbol. */
template <typename Model, size_t ReservoirSize>
inline void ReservoirWrapper<Model, ReservoirSize>::update(bit_t b) {

    m_model.update(b);
    m_reservoir.observe(b);
}


/* File extension. */
template <typename Model, size_t ReservoirSize>
inline const char *ReservoirWrapper<Model, ReservoirSize>::fileExtension() const {

    return m_model.fileExtension();
}


/* Make a copy of the model. */
template <typename Model, size_t ReservoirSize>
inline Compressor *ReservoirWrapper<Model, ReservoirSize>::clone() const {

    return new ReservoirWrapper<Model, ReservoirSize>(*this);
}


/* Wrapped reservoir accessor. */
template <typename Model, size_t ReservoirSize>
inline const ReservoirSampling<bit_t> &ReservoirWrapper<Model, ReservoirSize>::reservoir() const {

    return m_reservoir;
}


#endif // __RESERVOIR_WRAPPER_HPP__

