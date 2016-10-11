#include "model_pool.hpp"

#include "mixture.hpp"

#include <map>
#include <stdexcept>
#include <algorithm>


static boost::uuids::random_generator uuid_gen;


/* helper routine for sorting by score. */
static bool modelEntryScoreComp(
    const ModelPoolEntry &lhs,
    const ModelPoolEntry &rhs
) {
    return lhs.score < rhs.score;
}


/* construct a model pool with a fixed capacity. */
ModelPool::ModelPool(size_t capacity) :
    m_capacity(capacity)
{
}


/* how many models are currently in the pool */
size_t ModelPool::size() const {

    return m_models.size();
}


/* how many models are allowed in the pool. */
size_t ModelPool::capacity() const {

    return m_capacity;
}


/* additively update a models score. */
void ModelPool::updateScore(const model_id_t &id, double delta) {

    ModelPoolEntry *e = findHelper(id);
    if (e == nullptr)
        throw std::runtime_error("ModelPool::updateScore : no matching id");

    e->score += delta;
}


/* retrieve the counterfactual score for a model. */
double ModelPool::score(const model_id_t &id) {

    ModelPoolEntry *e = findHelper(id);
    if (e == nullptr)
        throw std::runtime_error("ModelPool::score : no matching id");

    return e->score;
}


/* find the model with a given id, nullptr on failure. */
ModelPoolEntry *ModelPool::findHelper(const model_id_t &id) {

    for (size_t i=0; i < m_models.size(); i++) {
        if (m_models[i].id == id)
            return &m_models[i];
    }

    return nullptr;
}


/* find the model with a given id, nullptr on failure. */
const ModelPoolEntry *ModelPool::find(const model_id_t &id) const {

    for (size_t i = 0; i < m_models.size(); i++) {
        if (m_models[i].id == id)
            return &m_models[i];
    }

    return nullptr;
}


/* adds a model state to the pool, taking ownership of the memory. */
void ModelPool::add(Compressor *state, double score, size_t time_added) {

    m_models.push_back(ModelPoolEntry());

    m_models.back().model = state;
    m_models.back().id    = uuid_gen();
    m_models.back().score = score;
    m_models.back().age   = time_added;
}


/* does a particular model exist within the mixture? */
bool ModelPool::exists(const model_id_t &id) const {

    for (size_t i=0; i < m_models.size(); i++) {
        if (m_models[i].id == id)
            return true;
    }

    return false;
}


/* removes the lowest scoring models until the pool
   no longer exceeds its capacity. implementation is
   currently quadratic complexity in the size of the modelpool,
   but this can be reduced using better data structures, but
   probably overkill given that the algorithm typically is used
   with a small capacity. */
void ModelPool::enforceCapacity() {

    // ensure that there are no duplicates
    bool duplicates = false;
    std::map<model_id_t,int> seen;
    for (size_t i=0; i < m_models.size(); i++) {
        seen[m_models[i].id]++;
        if (seen[m_models[i].id] > 1) {
            duplicates = true;
            break;
        }
    }
    assert(!duplicates);

    while (m_models.size() > m_capacity) {
        auto it = std::min_element(
            m_models.begin(),
            m_models.end(),
            modelEntryScoreComp
        );

        // todo: reclaim memory for the saved model state?

        m_models.erase(it);
    }
}


/* create a new mixture containing all of the models that
   currently reside in the model pool plus a new instance
   of the base model. ownership of the mixture resides with
   the caller. */
Mixture *ModelPool::createMixture(const Compressor *base_model) const {

    id_list_t ids;

    Mixture::modelclass_t modelpool;
    for (size_t i=0; i < m_models.size(); i++) {
        modelpool.push_back(m_models[i].model->clone());
        ids.push_back(m_models[i].id);
    }

    // only create a new base model if it is requested
    if (base_model != nullptr) {
        modelpool.push_back(base_model->clone());
        ids.push_back(uuid_gen());  // generate a new id for the reset model
    }

    Mixture::prior_t prior;
    if (base_model != nullptr) {
        prior.assign(modelpool.size() - 1, 0.5 / (modelpool.size() - 1));
        prior.push_back(modelpool.size() == 1 ? 1.0 : 0.5);
    } else {
        prior.assign(modelpool.size(), 1.0 / modelpool.size());
    }

    assert(prior.size() == ids.size());
    assert(modelpool.size() > 0);

    return new Mixture(modelpool, prior, ids);
}


/* model entry accessor, returns nullptr if out of bounds index used. */
const ModelPoolEntry *ModelPool::modelEntry(size_t idx) const {

    if (idx >= m_models.size()) return nullptr;

    return &m_models[idx];
}


/* removes a model from the modelpool. */
void ModelPool::remove(const model_id_t &id) {

    m_models.erase(
        std::remove_if(
            m_models.begin(),
            m_models.end(),
            [&](const ModelPoolEntry &lhs) {
                return lhs.id == id;
            }
        ),
        m_models.end()
    );
}


/* removes the oldest models until the capacity is no longer exceeded. */
size_t ModelPool::pruneOldest() {

    // skip pruning if capacity is not exceeded
    if (size() <= capacity()) return 0;

    size_t rval = 0;

    // get a sorted list of ages
    std::vector<size_t> ages;
    for (size_t i = 0; i < m_models.size(); i++) {
        ages.push_back(m_models[i].age);
    }

    std::sort(ages.begin(), ages.end(), std::greater<size_t>());
    size_t culling_age = ages[capacity()];

    // prune models with the smallest ages
    std::vector<model_id_t> to_remove;

    for (size_t i = 0; i < m_models.size(); i++) {
        if (m_models[i].age <= culling_age)
            to_remove.push_back(m_models[i].id);
    }

    for (auto idx : to_remove) {
        // note: this routine is O(n), doesn't need to be,
        //       but left this way for simplicity
        remove(idx);
    }

    rval += to_remove.size();
    assert(size() <= capacity());

    return rval;
}

