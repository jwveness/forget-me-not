#ifndef __STATCOUNTER_HPP__
#define __STATCOUNTER_HPP__

#include <cassert>

/*
 * Stores summary stats without keeping track of the actual seen numbers.
 *
 * Implements Knuth's online algorithm for variance, first one
 * found under "Online Algorithm" of
 * http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 *
 * Uses the incremental algorithm for updating the mean found at
 * http://webdocs.cs.ualberta.ca/~sutton/book/2/node6.html
 *
 */

class StatCounter {

    public:

        StatCounter();

        // reininitialise
        void reset();

        // add a sample
        void push(double num);

        // sample variance
        double variance() const;

        // sample mean
        double mean() const;

        // number of samples
        size_t size() const;

    private:

        double m_sum;
        double m_m2;
        double m_mean;
        size_t m_n;
};


inline StatCounter::StatCounter() {

    reset();
}


/* reininitialise */
inline void StatCounter::reset() {

    m_n = 0;
    m_sum = m_m2 = m_mean = 0.0;
}


/* add a sample */
inline void StatCounter::push(double num) {

  m_sum += num;
  m_n++;

  double delta = num - m_mean;
  m_mean += delta / m_n;
  m_m2 += delta*(num - m_mean);
}


/* sample variance */
inline double StatCounter::variance() const {

    return m_m2 / m_n;
}


/* sample mean */
inline double StatCounter::mean() const {

    return m_mean;
}


/* number of samples */
inline size_t StatCounter::size() const {

    return m_n;
}


#endif  // __STATCOUNTER_HPP__

