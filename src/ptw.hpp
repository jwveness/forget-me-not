#ifndef __PTW_HPP__
#define __PTW_HPP__

#include "common.hpp"
#include "kt.hpp"

#include <sstream>


// Partition Tree Weighting
template <typename BaseModel, size_t Depth>
class PTW : public Compressor {

    template <typename BaseNodeModel>
    struct PTWNode {

        PTWNode() :
            m_log_weighted(0.0),
            m_log_buf(0.0),
            m_model(new BaseNodeModel())
        { }

        ~PTWNode() {
            delete m_model;
        }

        // copy constructor
        PTWNode(const PTWNode<BaseNodeModel> &rhs) {
            copyWorker(rhs);
        }

        // assignment operator
        PTWNode<BaseNodeModel> &operator=(const PTWNode<BaseNodeModel> &rhs) {

            if (boost::addressof(*this) != boost::addressof(rhs)) {
                Compressor *old = m_model;
                copyWorker(rhs);
                delete old;
            }

            return *this;
        }

        // performs a deep copy
        void copyWorker(const PTWNode<BaseNodeModel> &rhs) {
            m_model = rhs.m_model->clone();
            m_log_weighted = rhs.m_log_weighted;
            m_log_buf = rhs.m_log_buf;
        }

        Compressor *m_model;
        double m_log_weighted;
        double m_log_buf;
    };

    typedef PTWNode<BaseModel> PTWnode_t;

    public:

        typedef uint64_t index_t;

        PTW();

        // the probability of seeing a bit next
        double prob(bit_t b);

        // the logarithm of the probability of all processed bits
        double logBlockProbability() const { return m_nodes[0].m_log_weighted; }

        // process a new bit
        void update(bit_t b);

        // file extension
        const char *fileExtension() const {

            static std::string name;
            if (name == "") {
                std::ostringstream oss;
                oss << "PTW-" << m_nodes[0].m_model->fileExtension();
                name = oss.str();
            }

            return name.c_str();
        }

        // create a clone from the current partition tree weighting mixture
        Compressor *clone() const {

            return new PTW<BaseModel,Depth>(*this);
        }

    private:

        // the number of bits to the left of the most significant
        // location at which times t-1 and t-2 differ, where t is
        // the 1 based representation of the current time
        size_t mscb(index_t t) const;

        index_t m_index;
        PTWnode_t m_nodes[Depth+1];
};


/* PTW constructor */
template <typename BaseModel, size_t Depth>
inline PTW<BaseModel, Depth>::PTW() :
    m_index(0)
{
    BOOST_STATIC_ASSERT(Depth <= 60);
}


/* the probability of seeing a particular symbol next */
template <typename BaseModel, size_t Depth>
inline double PTW<BaseModel, Depth>::prob(bit_t b) {

    // TODO: simple, but a bit slow...
    std::auto_ptr<Compressor> x(clone());
    x->update(b);

    return std::exp(x->logBlockProbability() - logBlockProbability());
}


/* process a new piece of sensory experience */
template <typename BaseModel, size_t Depth>
inline void PTW<BaseModel, Depth>::update(bit_t b) {

    assert(m_index < (1 << Depth));

    // precompute logarithms
    static const double LogSplitWeight = std::log(0.5);
    static const double LogStopWeight =  std::log(0.5);

    // mscb requires the current 1-based time
    size_t i = mscb(m_index + 1);

    // save weighted probability in change point's parent
    m_nodes[i].m_log_buf = m_nodes[i+1].m_log_weighted;

    // now reset statistics from the change point downwards
    for (size_t j=i+1; j <= Depth; j++) {
        m_nodes[j] = PTWnode_t();
    }

    // compute weighted probability from bottom up
    PTWnode_t &n = m_nodes[Depth];
    n.m_model->update(b);
    n.m_log_weighted = n.m_model->logBlockProbability();

    for (size_t i=1; i <= Depth; i++) {
        size_t idx = Depth - i;
        m_nodes[idx].m_model->update(b);
        double lhs = LogStopWeight + m_nodes[idx].m_model->logBlockProbability();
        double rhs = LogSplitWeight + m_nodes[idx+1].m_log_weighted + m_nodes[idx].m_log_buf;
        m_nodes[idx].m_log_weighted = logAdd(lhs, rhs);
    }

    m_index++;
}


/* the number of bits to the left of the most significant
   location at which times t-1 and t-2 differ, where t is
   the 1 based current time. */
template <typename BaseModel, size_t Depth>
inline size_t PTW<BaseModel, Depth>::mscb(index_t t) const {

    if (t == 1) return 0;

    size_t c = Depth-1;
    size_t cnt = 0;

    for (index_t i = 0; i < Depth; i++) {

        index_t tm1 = t - 1, tm2 = t-2;
        index_t mask = 1 << c;

        if ((tm1 & mask) != (tm2 & mask)) return cnt;

        c--, cnt++;
    }

    return cnt;
}


#endif // __PTW_HPP__


