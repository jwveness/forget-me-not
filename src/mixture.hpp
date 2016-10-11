#ifndef __MIXTURE_HPP__
#define __MIXTURE_HPP__

#include <sstream>

#include "common.hpp"

// Boost includes
#include <boost/ptr_container/ptr_vector.hpp>


typedef std::vector<model_id_t> id_list_t;


// a finite bayes mixture
class Mixture : public Compressor {

    public:

        typedef std::vector<double> prior_t;
        typedef std::vector<Compressor *> modelclass_t;

        // construct an unlabelled mixture of models
        Mixture(const modelclass_t &models, const std::vector<double> &prior);

        // construct a labelled mixture of models
        Mixture(const modelclass_t &models,
            const std::vector<double> &prior, const id_list_t &ids);

        // copy constructor
        Mixture(const Mixture &rhs);

        // assignment operator
        Mixture &operator=(const Mixture &rhs);

        // the probability of seeing a bit next
        double prob(bit_t b);

        // the logarithm of the probability of all processed bits
        double logBlockProbability() const;

        // the log liklihood of the i'th model
        double logLiklihood(size_t i) const;

        // posterior probability of the i'th model
        double posterior(size_t i) const;

        // the id of the i'th model in the mixture, nullptr if we have
        // either an out of bounds index or a unlabelled mixture.
        const model_id_t *modelId(size_t i) const;

        // process a new bit
        void update(bit_t b);

        // file extension
        const char *fileExtension() const;

        // model accessor
        const modelclass_t &models() const;

        // destructor
        ~Mixture();

        // create a clone of the mixture model and the state of all of its submodels
        Compressor *clone() const;

        // how many examples has each model in the mixture seen?
        size_t nExamples() const;

    protected:

        // copy constructor worker
        void copyWorker(const Mixture &rhs);

        Mixture() { }

        modelclass_t m_models;
        std::vector<double> m_log_prior_liklihood;
        std::vector<double> m_log_liklihood;
        double m_lbp;
        size_t m_seen;
        id_list_t m_ids;
};


#endif // __MIXTURE_HPP__


