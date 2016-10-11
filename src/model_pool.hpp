#ifndef __MODEL_POOL_HPP__
#define __MODEL_POOL_HPP__

#include "common.hpp"

#include <vector>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/scoped_ptr.hpp>


class Mixture;


/* Holds the information relevant to a particular model. */
struct ModelPoolEntry {

    Compressor  *model;
    model_id_t  id;
    double      score;
    size_t      age;
};


/* A class which manages a pool of saved model states, and their
   corresponding priority scores. A pool has a target capacity,
   which can be enforced via enforceCapacity(). */
class ModelPool {

    public:

        // construct a model pool with a desired capacity
        ModelPool(size_t capacity);

        // how many models are currently in the pool
        size_t size() const;

        // what is the maximum capacity of the pool
        size_t capacity() const;

        // additively update a models score
        void updateScore(const model_id_t &id, double delta);

        // retrieve the score for a model
        double score(const model_id_t &id);

        // adds a model state to the pool, taking ownership of it
        void add(Compressor *state, double score, size_t time_added);

        // does a particular model exist within the mixture
        bool exists(const model_id_t &id) const;

        // removes the lowest scoring models until the pool
        // no longer exceeds its capacity
        void enforceCapacity();

        // create a new mixture containing all of the models that
        // currently reside in the model pool plus a new instance
        // of the base model if base_model != nullptr. ownership of
        // the mixture resides with the caller.
        Mixture *createMixture(const Compressor *base_model) const;

        // model entry accessor, returns nullptr if out of bounds index used.
        const ModelPoolEntry *modelEntry(size_t idx) const;

        // removes a model from the modelpool
        void remove(const model_id_t &id);

        // find the model with a given id
        const ModelPoolEntry *find(const model_id_t &id) const;

        // removes the oldest models until the capacity is no longer exceeded,
        // returning the number of pruned models
        size_t pruneOldest();

    private:

        // find the model with a given id
        ModelPoolEntry *findHelper(const model_id_t &id);

        std::vector<ModelPoolEntry> m_models;
        size_t m_capacity;
};


#endif // __MODEL_POOL_HPP__


