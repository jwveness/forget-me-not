#ifndef __RESERVOIR_SAMPLING_HPP__
#define __RESERVOIR_SAMPLING_HPP__

#include <vector>
#include <utility>

#include <boost/random.hpp>
#include <boost/random/uniform_int.hpp>


// An implementation of the Reservoir Sampling, which given
// a stream of data >= n, returns a uniform sample of size k
// without replacement
template <typename Symbol>
class ReservoirSampling {

    public:

        typedef std::vector<std::pair<Symbol,size_t>> sample_t;

        // construct a reservoir which returns a sample of size k.
        ReservoirSampling(size_t k);

        // sample accessor.
        const sample_t &sample() const;

        // observe a new symbol.
        void observe(const Symbol &sym);

        // how many symbols has the reservoir seen?
        size_t seen() const;

        // number of symbols inside the reservoir,
        // which is at most k.
        size_t size() const;

        // clears the reservoir back to an empty state
        void reset();

    private:

        sample_t m_sample;
        size_t m_k;
        size_t m_seen;

        // random number source
        boost::mt19937 m_randsrc;
        boost::uniform_01<boost::mt19937 &> m_urng;
};


/* construct a reservoir which returns a sample of size k. */
template <typename Symbol>
ReservoirSampling<Symbol>::ReservoirSampling(size_t k) :
    m_k(k),
    m_seen(0),
    m_randsrc(666),
    m_urng(m_randsrc)
{
}


/* sample accessor. */
template <typename Symbol>
const typename ReservoirSampling<Symbol>::sample_t &ReservoirSampling<Symbol>::sample() const {

    return m_sample;
}


/* observe a new symbol. */
template <typename Symbol>
void ReservoirSampling<Symbol>::observe(const Symbol &sym) {

    m_seen++;

    if (m_sample.size() < m_k) {
        m_sample.push_back(std::make_pair(sym, m_seen));
        return;
    }

    assert(m_sample.size() == m_k);

    // accept the new symbol as part of the reservoir
    // with probability k / n.
    double n = static_cast<double>(m_seen);
    size_t idx = static_cast<size_t>(m_urng() * n);
    if (idx < m_k)
        m_sample[idx] = std::make_pair(sym, m_seen);
}


/* how many symbols has the reservoir seen? */
template <typename Symbol>
size_t ReservoirSampling<Symbol>::seen() const {

    return m_seen;
}


/* number of symbols inside the reservoir, which is at most k. */
template <typename Symbol>
size_t ReservoirSampling<Symbol>::size() const {

    return m_sample.size();
}


/* clears the reservoir back to an empty state. */
template <typename Symbol>
void ReservoirSampling<Symbol>::reset() {

    m_sample.clear();
    m_seen = 0;
}


#endif // __RESERVOIR_SAMPLING_HPP__

