#ifndef __FMN_HPP__
#define __FMN_HPP__

#include <iomanip>
#include <map>
#include <algorithm>

#include "common.hpp"
#include "model_pool.hpp"
#include "reservoir_sampling.hpp"
#include "reservoir_wrapper.hpp"



// do we do an additional check to avoid adding a useless model to the modelpool
constexpr bool UseRedundantAddPruning = true;

// do we enforce the capacity constraints of the modelpool
constexpr bool UseCapacityEnforcement = true;

// at what level do we consider adding models into the pool
constexpr size_t ModelAdditionThreshold = 5;

// do we only consider adding/replacing models on MAP segments?
constexpr bool UseReplaceOnlyOnMAPSegments = false;
constexpr bool UseAddOnlyOnMAPSegments     = false;

// global reservoir size
constexpr size_t GlobalReservoirSize = 100;


// hyperparameters which govern the prior assigned to each
// distinct hierarchical segmentation
const double PruningFMNLogSplitWeight = std::log(0.5);
const double PruningFMNLogStopWeight  = std::log(0.5);



// Resource Bounded  - Forget Me Not distribution
template <typename BaseModel, size_t Depth>
class PruningForgetMeNot : public Compressor {

    typedef boost::ptr_vector<Compressor> modelpool_t;
    typedef std::vector<double> modelscores_t;

    struct Stats {
        size_t adds          = 0;
        size_t replaces      = 0;
        size_t ignores       = 0;
        size_t fresh_adds    = 0;
        size_t aborted_adds  = 0;
        size_t map_ignores   = 0;
        size_t pruned_models = 0;
    };

    struct PruningForgetMeNotNode {

        PruningForgetMeNotNode() :
            m_model(nullptr),
            m_log_weighted(0.0),
            m_log_buf(0.0),
            m_reservoir(BaseModel::ReservoirSlots)
        { }

        void reset() {
            m_model               = nullptr;
            m_log_weighted        = 0.0;
            m_log_buf             = 0.0,
            m_reservoir.reset();
        }

        Compressor *m_model;
        double m_log_weighted;
        double m_log_buf;
        ReservoirSampling<bit_t> m_reservoir;
    };

    typedef PruningForgetMeNotNode FMNnode_t;

    public:

        typedef uint64_t index_t;

        // construct a memory bounded forget me not, with k model slots
        PruningForgetMeNot(size_t k);

        // the probability of seeing a bit next
        double prob(bit_t b);

        // the logarithm of the probability of all processed bits
        double logBlockProbability() const;

        // process a new bit
        void update(bit_t b);

        // file extension
        const char *fileExtension() const;

        // clones the forget me not model
        Compressor *clone() const;

        // model pool accessor
        const ModelPool &modelPool() const;

        // get a string descriptor of the saved model properties
        std::string modelInfo(size_t idx);

        // write out a textual description of the saved model states
        std::ostream &poolInfo(std::ostream &o, size_t seqlength, size_t window);

        // get a string descriptor of the process
        std::string getSummary() const;

    private:

        // the number of bits to the left of the most significant
        // location at which times t-1 and t-2 differ, where t is
        // the 1 based representation of the current time
        size_t mscb(index_t t) const;

        // compute the posterior weights for current temporal discretization level
        void levelPosterior(std::vector<double> &dest) const;

        // compute the MAP level in the temporal hierachy, breaking ties in favour
        // of the most coarse segmentation
        size_t mapLevel() const;

        // save the model states worth remembering from all of the levels in
        // hierachy below and including 'from'.
        void saveClosedSegmentModels(size_t from);

        // find the highest probability segment starting from (inclusively) a
        // given depth, with ties broken in favour of the most coarse segmentation
        size_t mostProbableLevelFrom(size_t d) const;

        // determine what to do with the state of of a mixture once a segment closes
        void applySegmentCloseOp(size_t depth, const Mixture &m);

        // check to see if a winning model would add no value with respect to the current mixture
        bool isRedundantWinner(const Mixture &m, size_t bestidx, size_t depth) const;

        // enforce the memory constraints on the size of the modelpool, returning
        // the number of pruned models
        size_t enforceCapacity();

        index_t m_index;
        FMNnode_t m_nodes[Depth + 1];
        ModelPool m_modelpool;
        Compressor *m_base_model;
        ReservoirSampling<bit_t> m_reservoir;
        Stats m_stats;
};


/* Forget-Me-Not constructor. */
template <typename BaseModel, size_t Depth>
inline PruningForgetMeNot<BaseModel, Depth>::PruningForgetMeNot(size_t k) :
    m_index(0),
    m_modelpool(k),
    m_base_model(new BaseModel()),
    m_reservoir(GlobalReservoirSize)
{
    BOOST_STATIC_ASSERT(Depth <= 60);

    Mixture *parent_mixture = m_modelpool.createMixture(m_base_model);
    for (size_t i = 0; i <= Depth; i++) {
        m_nodes[i].m_model = parent_mixture;
    }
}


/* The log-marginal probability of the data. */
template <typename BaseModel, size_t Depth>
inline double PruningForgetMeNot<BaseModel, Depth>::logBlockProbability() const {

    return m_nodes[0].m_log_weighted;
}


/* The estimated probability of seeing a particular symbol next. */
template <typename BaseModel, size_t Depth>
double PruningForgetMeNot<BaseModel, Depth>::prob(bit_t b) {

    // compute the posterior probability of each level in the hierachy
    std::vector<double> level_posterior;
    levelPosterior(level_posterior);

    // compute the predictive probability
    double rval = 0.0, p;
    std::map<Compressor *, double> cache;

    for (size_t i = 0; i <= Depth; i++) {

        Compressor *m = m_nodes[i].m_model;

        // hit the cache to avoid recomputing
        // unnecessary probabilities
        auto it = cache.find(m);
        if (it == cache.end()) {
            p = m->prob(b);
            cache[m] = p;
        } else {
            p = it->second;
        }

        rval += level_posterior[i] * p;
    }

    return rval;
}


/* Would a mixture of the existing modelpool state do better than
   the current winning model on the subset of data it has been trained on?*/
template <typename BaseModel, size_t Depth>
bool PruningForgetMeNot<BaseModel, Depth>::isRedundantWinner(
    const Mixture &m,
    size_t bestidx,
    size_t depth
) const {

    if (!UseRedundantAddPruning) return false;

    if (m_modelpool.size() == 0) return false;

    std::unique_ptr<Mixture> tmp_m(m_modelpool.createMixture(nullptr));
    Compressor *c = const_cast<Compressor *>(m.models()[bestidx]);
    const ReservoirSampling<bit_t>::sample_t &data = m_nodes[depth].m_reservoir.sample();

    double log_mixture = 0.0, log_winner = 0.0;

    for (size_t i = 0; i < data.size(); i++) {
        log_mixture += std::log(tmp_m->prob(data[i].first));
        log_winner  += std::log(c->prob(data[i].first));
    }
    return log_mixture > log_winner;
}


/* Determine what the do with the models in the mixture of the closing segment. */
template <typename BaseModel, size_t Depth>
void PruningForgetMeNot<BaseModel, Depth>::applySegmentCloseOp(size_t depth, const Mixture &m) {

    assert(!m.models().empty());

    // find the best performing model in the current segment
    double bestprob = -std::numeric_limits<double>::max();
    size_t bestidx = 0;

    for (size_t i = 0; i < m.models().size(); i++) {
        if (m.logLiklihood(i) > bestprob) {
            bestprob = m.logLiklihood(i);
            bestidx = i;
        }
    }

    // is the winning model a freshly created model?
    bool fresh_winner = bestidx == m.models().size() - 1;

    // if the winning model is better at encoding the
    // parent models data, then replace the parent with the new model
    const Compressor *c = m.models()[bestidx];
    Compressor *lo = const_cast<Compressor *>(c);

    const model_id_t *id = m.modelId(bestidx);
    const ModelPoolEntry *mpe = id != nullptr ? m_modelpool.find(*id) : nullptr;

    if (!fresh_winner && id != nullptr && mpe != nullptr) {

        Compressor *pa = const_cast<Compressor *>(mpe->model);
        const BaseModel *rw_pa = static_cast<const BaseModel *>(mpe->model);
        const ReservoirSampling<bit_t>::sample_t &data = rw_pa->reservoir().sample();
        double parent_log_prob = 0.0, winner_log_prob = 0.0;

        for (size_t i = 0; i < data.size(); i++) {
            winner_log_prob += std::log(lo->prob(data[i].first));
            parent_log_prob += std::log(pa->prob(data[i].first));
        }

        // if the current model is better at encoding the data
        // inside the parent model, then replace
        if (winner_log_prob > parent_log_prob &&
            (!UseReplaceOnlyOnMAPSegments || depth == mapLevel())
        ) {
            m_stats.replaces++;
            m_modelpool.add(lo->clone(), 0.0, m_index);
            m_modelpool.remove(*id);
            return;
        }
    }

    if (!UseAddOnlyOnMAPSegments || depth == mapLevel()) {

        // check to see if the model actually improves performance on the
        // current segment in hindsight with respect to the current mixture state
        if (isRedundantWinner(m, bestidx, depth)) {
            m_stats.aborted_adds++;
            m_stats.ignores++;
            return;
        }

        // add model to modelpool
        if (fresh_winner) m_stats.fresh_adds++;
        m_stats.adds++;
        m_modelpool.add(m.models()[bestidx]->clone(), 0.0, m_index);
        return;
    }

    // otherwise ignore the data in the segment
    m_stats.map_ignores++;
    m_stats.ignores++;
}


/* Save the model states worth remembering from all of the levels in
   hierachy below and including 'from'. */
template <typename BaseModel, size_t Depth>
void PruningForgetMeNot<BaseModel, Depth>::saveClosedSegmentModels(size_t from) {

    // we have a number of segments which are closing
    for (size_t j = from; j <= Depth; j++) {

        Mixture *mix_ptr = dynamic_cast<Mixture *>(m_nodes[j].m_model);

        if (m_nodes[j].m_model != nullptr &&
            mix_ptr->nExamples() > 0      &&
            Depth - j >= ModelAdditionThreshold
        ) {
            applySegmentCloseOp(j, *mix_ptr);
        }
    }
}


/* Process a new symbol. */
template <typename BaseModel, size_t Depth>
inline void PruningForgetMeNot<BaseModel, Depth>::update(bit_t b) {

    assert(m_index < (1 << Depth));

    // mscb requires the current 1-based time
    size_t i = mscb(m_index + 1);

    // save weighted probability in change point's parent
    m_nodes[i].m_log_buf = m_nodes[i + 1].m_log_weighted;

    // to ensure we never reclaim the memory for a parent
    std::set<Compressor *> skip_delete;
    for (size_t j = 0; j < i + 1; j++)
        skip_delete.insert(m_nodes[j].m_model);

    // save the models for all relevant closed segments
    saveClosedSegmentModels(i + 1);

    // enforce the model pool capacity
    size_t n_pruned = enforceCapacity();
    m_stats.pruned_models += n_pruned;

    // now reset statistics from the change point downwards
    Mixture *parent_mixture = m_modelpool.createMixture(m_base_model);

    for (size_t j = i + 1; j <= Depth; j++) {

        // release the memory for the old mixture
        Compressor *p = m_nodes[j].m_model;
        if (p != nullptr && skip_delete.find(p) == skip_delete.end()) {
            delete p;
            skip_delete.insert(p);
        }

        m_nodes[j].reset();
        m_nodes[j].m_model = parent_mixture;
    }

    // update the base models, being careful to only update
    // each distinct mixture instance once
    std::set<Compressor *> seen;
    for (size_t i = 0; i <= Depth; i++) {
        Compressor *p = m_nodes[i].m_model;
        if (seen.find(p) == seen.end()) {
            p->update(b);
            seen.insert(p);
        }
        m_nodes[i].m_reservoir.observe(b);
    }

    // compute weighted probability from bottom up
    FMNnode_t &n = m_nodes[Depth];
    n.m_log_weighted = n.m_model->logBlockProbability();

    for (size_t i = 1; i <= Depth; i++) {
        size_t idx = Depth - i;
        double lhs = PruningFMNLogStopWeight + m_nodes[idx].m_model->logBlockProbability();
        double rhs = PruningFMNLogSplitWeight + m_nodes[idx + 1].m_log_weighted + m_nodes[idx].m_log_buf;
        m_nodes[idx].m_log_weighted = logAdd(lhs, rhs);
    }

    m_index++;
    m_reservoir.observe(b);
}


/* Compute the posterior weights for current temporal discretization level. */
template <typename BaseModel, size_t Depth>
void PruningForgetMeNot<BaseModel, Depth>::levelPosterior(std::vector<double> &dest) const {

    double posterior_mass_left = 1.0;

    dest.clear();

    // compute the posterior weights of each level from top down
    for (size_t i = 0; i <= Depth; i++) {

        // compute log posterior of stopping at level i
        double x = PruningFMNLogStopWeight + m_nodes[i].m_model->logBlockProbability();
        x -= m_nodes[i].m_log_weighted;
        double stop_post = std::exp(x);

        dest.push_back(posterior_mass_left * stop_post);
        posterior_mass_left *= (1.0 - stop_post);

        assert(dest.back() >= 0.0 && dest.back() <= 1.0);

        // for numerical stability
        posterior_mass_left = std::max(posterior_mass_left, 0.0);
        assert(posterior_mass_left >= 0.0 && posterior_mass_left <= 1.0);
    }

    assert(dest.size() == Depth + 1);
}


/* Create a new instance of this model. */
template <typename BaseModel, size_t Depth>
inline Compressor *PruningForgetMeNot<BaseModel, Depth>::clone() const {

    // dummy routine, not used
    return nullptr;
}


/* Modelpool accessor. */
template <typename BaseModel, size_t Depth>
inline const ModelPool &PruningForgetMeNot<BaseModel, Depth>::modelPool() const {

    return m_modelpool;
}


/* Compute the MAP level in the temporal hierachy, breaking ties in favour
   of the most coarse segmentation. */
template <typename BaseModel, size_t Depth>
inline size_t PruningForgetMeNot<BaseModel, Depth>::mapLevel() const {

    return mostProbableLevelFrom(0);
}


/* Find the highest probability segment starting from (inclusively) a
   given depth, with ties broken in favour of the most coarse segmentation. */
template <typename BaseModel, size_t Depth>
inline size_t PruningForgetMeNot<BaseModel, Depth>::mostProbableLevelFrom(size_t d) const {

    assert(d <= Depth);

    std::vector<double> post;
    levelPosterior(post);

    auto it = std::max_element(post.begin() + d, post.end());
    return d + std::distance(post.begin() + d, it);
}


/* The number of bits to the left of the most significant
   location at which times t-1 and t-2 differ, where t is
   the 1 based current time. */
template <typename BaseModel, size_t Depth>
inline size_t PruningForgetMeNot<BaseModel, Depth>::mscb(index_t t) const {

    if (t == 1) return 0;

    size_t c = Depth - 1;
    size_t cnt = 0;

    for (index_t i = 0; i < Depth; i++) {

        index_t tm1 = t - 1, tm2 = t - 2;
        index_t mask = 1 << c;

        if ((tm1 & mask) != (tm2 & mask)) return cnt;

        c--, cnt++;
    }

    return cnt;
}


/* Enforce the memory constraints on the size of the modelpool,
   returning the number of pruned models. */
template <typename BaseModel, size_t Depth>
size_t PruningForgetMeNot<BaseModel, Depth>::enforceCapacity() {

    if (!UseCapacityEnforcement) return 0;

    return m_modelpool.pruneOldest();
}


/* Determine the file extension. */
template <typename BaseModel, size_t Depth>
const char *PruningForgetMeNot<BaseModel, Depth>::fileExtension() const {

    static std::string name;

    if (name == "") {
        std::ostringstream oss;
        const Mixture *p = dynamic_cast<const Mixture *>(m_nodes[0].m_model);
        oss << "FMN(" << p->models()[0]->fileExtension() << ")";
        name = oss.str();
    }

    return name.c_str();
}


/* Get a string description of the saved model properties. */
template <typename BaseModel, size_t Depth>
inline std::string PruningForgetMeNot<BaseModel, Depth>::modelInfo(size_t idx) {

    // TODO:
    return "";
}


/* Write out a textual description of the model pool to a stream. */
template <typename BaseModel, size_t Depth>
std::ostream &PruningForgetMeNot<BaseModel, Depth>::poolInfo(
    std::ostream &o,
    size_t seqlength,
    size_t window
) {

    for (size_t i = 0; i < m_modelpool.size(); i++) {

        const ModelPoolEntry *e = m_modelpool.modelEntry(i);

        TrackingKT *p = dynamic_cast<TrackingKT *>(e->model);
        o << "\t";

        if (p != nullptr) {
            p->displayIndices(seqlength, window);
        }

        o << std::endl;
    }

    return o;
}


/* Get a string summarizing some of the process statistics. */
template <typename BaseModel, size_t Depth>
std::string PruningForgetMeNot<BaseModel, Depth>::getSummary() const {

    std::ostringstream oss;

    // output some summary statistics
    oss << "aborted_adds: "  << m_stats.aborted_adds << ", ";
    oss << "fresh_adds: "    << m_stats.fresh_adds << ", ";
    oss << "adds: "          << m_stats.adds << ", ";
    oss << "replaces: "      << m_stats.replaces << ", ";
    oss << "ignores: "       << m_stats.ignores << ", ";
    oss << "map_ignores: "   << m_stats.map_ignores << ", ";
    oss << "pruned_models: " << m_stats.pruned_models;

    return oss.str();
}


#endif // __FMN_HPP__

