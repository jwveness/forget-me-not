#include "mixture.hpp"


/* Constructor which creates an unlabelled mixture. */
Mixture::Mixture(const modelclass_t &models, const std::vector<double> &prior) :
    m_models(models),
    m_lbp(0.0),
    m_seen(0),
    m_log_liklihood(models.size(), 0.0)
{
    for (size_t i=0; i < models.size(); i++) {
        m_log_prior_liklihood.push_back(std::log(prior[i]));
    }

    assert(m_ids.empty());
}


/* construct a labelled mixture of models. */
Mixture::Mixture(
    const modelclass_t &models,
    const std::vector<double> &prior,
    const id_list_t &ids
) :
    m_models(models),
    m_lbp(0.0),
    m_seen(0),
    m_log_liklihood(models.size(), 0.0)
{

    for (size_t i=0; i < models.size(); i++) {
        m_log_prior_liklihood.push_back(std::log(prior[i]));
        m_ids.push_back(ids[i]);
    }
}


/* copy constructor. */
Mixture::Mixture(const Mixture &rhs) {

    copyWorker(rhs);
}


/* assignment operator. */
Mixture &Mixture::operator=(const Mixture &rhs) {

    if (boost::addressof(*this) != boost::addressof(rhs)) {

        std::vector<Compressor *> old_models(rhs.m_models.begin(), rhs.m_models.end());
        copyWorker(rhs);
        for (size_t i=0; i < m_models.size(); i++)
            delete old_models[i];
    }

    return *this;
}


/* the probability of seeing a bit next. */
double Mixture::prob(bit_t b) {

    double rval = 0.0;
    for (size_t i=0; i < m_models.size(); i++) {
        double weight = std::exp(m_log_prior_liklihood[i] - m_lbp);
        rval += weight * m_models[i]->prob(b);
    }

    return rval;
}


/* the logarithm of the probability of all processed bits. */
double Mixture::logBlockProbability() const {

    return m_lbp;
}


/* the log liklihood of the i'th model */
double Mixture::logLiklihood(size_t i) const {

    assert(i < m_models.size());
    assert(m_log_liklihood[i] >= m_models[i]->logBlockProbability());

    return m_log_liklihood[i];
}


// posterior probability of the i'th model
double Mixture::posterior(size_t i) const {

    return std::exp(m_log_prior_liklihood[i] - m_lbp);
}


/* the id of the i'th model in the mixture, nullptr if we have
   either an out of bounds index or a unlabelled mixture. */
const model_id_t *Mixture::modelId(size_t i) const {

    if (m_ids.empty()) return nullptr;
    if (i >= m_ids.size()) return nullptr;

    return &m_ids[i];
}


/* process a new bit. */
void Mixture::update(bit_t b) {

    double newlbp;

    for (size_t i=0; i < m_models.size(); i++) {
        double log_p = std::log(m_models[i]->prob(b));
        m_log_prior_liklihood[i] += log_p;
        m_log_liklihood[i] += log_p;
        m_models[i]->update(b);
        newlbp = (i > 0) ? logAdd(newlbp, m_log_prior_liklihood[i]) : m_log_prior_liklihood[i];
    }

    m_lbp = newlbp;
    m_seen++;
}


/* file extension. */
const char *Mixture::fileExtension() const {

    static std::string name;

    std::ostringstream oss;
    oss << "Mixture(";
    for (size_t i=0; i < m_models.size(); i++) {
        if (i > 0) oss << ",";
        oss << m_models[i]->fileExtension();
    }
    oss << ")";
    name = oss.str();

    return name.c_str();
}


/* model accessor. */
const Mixture::modelclass_t &Mixture::models() const {

    return m_models;
}


/* destructor. */
Mixture::~Mixture() {

    for (size_t i=0; i < m_models.size(); i++)
        delete m_models[i];
}


/* create a clone of the mixture model and the state of all of its submodels. */
Compressor *Mixture::clone() const {

    return new Mixture(*this);
}


/* how many examples has each model in the mixture seen? */
size_t Mixture::nExamples() const {

    return m_seen;
}


/* copy constructor worker */
void Mixture::copyWorker(const Mixture &rhs) {

    for (size_t i=0; i < rhs.m_models.size(); i++) {
        m_models.push_back(rhs.m_models[i]->clone());
        m_log_prior_liklihood.push_back(rhs.m_log_prior_liklihood[i]);
        m_log_liklihood.push_back(rhs.m_log_liklihood[i]);
    }
    m_lbp  = rhs.m_lbp;
    m_seen = rhs.m_seen;
}

