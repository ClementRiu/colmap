//
// Created by clementriu on 6/6/22.
//

#ifndef COLMAP_SRC_OPTIM_LRTSAC_H
#define COLMAP_SRC_OPTIM_LRTSAC_H

#include <cfloat>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

//#include "optim/ransac.h"
#include "optim/random_sampler.h"
#include "optim/support_measurement.h"
#include "util/alignment.h"
#include "util/logging.h"

namespace colmap {

/// Chi2 table: {{dof,p},F^{-1}(p)} with F the cumulative Chi2 distribution.
/// Used dof: 2+ (Line2P,Essential5P,PNP6P, Fundamental7P,Fundemental8P,9:???)
static std::map<std::pair<int, double>, double> chi2Table = {
    {{2 + 2, 0.90}, 7.779},   {{2 + 2, 0.91}, 8.043},
    {{2 + 2, 0.92}, 8.337},   {{2 + 2, 0.93}, 8.666},
    {{2 + 2, 0.94}, 9.044},   {{2 + 2, 0.95}, 9.488},
    {{2 + 2, 0.96}, 10.026},  {{2 + 2, 0.97}, 10.712},
    {{2 + 2, 0.98}, 11.668},  {{2 + 2, 0.99}, 13.277},
    {{5 + 2, 0.90}, 12.017},  {{5 + 2, 0.91}, 12.337},
    {{5 + 2, 0.92}, 12.691},  {{5 + 2, 0.93}, 13.088},
    {{5 + 2, 0.94}, 13.540},  {{5 + 2, 0.95}, 14.067},
    {{5 + 2, 0.96}, 14.703},  {{5 + 2, 0.97}, 15.509},
    {{5 + 2, 0.98}, 16.622},  {{5 + 2, 0.99}, 18.475},
    {{6 + 2, 0.90}, 13.3616}, {{6 + 2, 0.91}, 13.6975},
    {{6 + 2, 0.92}, 14.0684}, {{6 + 2, 0.93}, 14.4836},
    {{6 + 2, 0.94}, 14.9563}, {{6 + 2, 0.95}, 15.5073},
    {{6 + 2, 0.96}, 16.1708}, {{6 + 2, 0.97}, 17.0105},
    {{6 + 2, 0.98}, 18.1682}, {{6 + 2, 0.99}, 20.0902},
    {{7 + 2, 0.90}, 14.684},  {{7 + 2, 0.91}, 15.034},
    {{7 + 2, 0.92}, 15.421},  {{7 + 2, 0.93}, 15.854},
    {{7 + 2, 0.94}, 16.346},  {{7 + 2, 0.95}, 16.919},
    {{7 + 2, 0.96}, 17.608},  {{7 + 2, 0.97}, 18.480},
    {{7 + 2, 0.98}, 19.679},  {{7 + 2, 0.99}, 21.666},
    {{8 + 2, 0.90}, 15.987},  {{8 + 2, 0.91}, 16.352},
    {{8 + 2, 0.92}, 16.753},  {{8 + 2, 0.93}, 17.203},
    {{8 + 2, 0.94}, 17.713},  {{8 + 2, 0.95}, 18.307},
    {{8 + 2, 0.96}, 19.021},  {{8 + 2, 0.97}, 19.922},
    {{8 + 2, 0.98}, 21.161},  {{8 + 2, 0.99}, 23.209},
    {{9 + 2, 0.90}, 17.275},  {{9 + 2, 0.91}, 17.653},
    {{9 + 2, 0.92}, 18.069},  {{9 + 2, 0.93}, 18.533},
    {{9 + 2, 0.94}, 19.061},  {{9 + 2, 0.95}, 19.675},
    {{9 + 2, 0.96}, 20.412},  {{9 + 2, 0.97}, 21.342},
    {{9 + 2, 0.98}, 22.618},  {{9 + 2, 0.99}, 24.725}};

struct LRTSACOptions {
  // Maximum error possible for the range of considered sigmas.
  double sigmaMax = 0.0;

  // A priori assumed minimum inlier ratio, which determines the maximum number
  // of iterations. Only applies if smaller than `max_num_trials`.
  double min_inlier_ratio = 0.1;

  // Confidence requiered to validate the run.
  double confidenceI = 0.0;
  // Increase the number of iterations to account for the early bailout .
  double confidenceIIB = 0.95;
  // Abort the iteration if minimum probability that one sample is free from
  // outliers is reached.
  double confidenceIIT = 0.99;

  // Improve the set of considered sigmas during computation.
  bool reduceSigma = true;

  // The num_trials_multiplier to the dynamically computed maximum number of
  // iterations based on the specified confidence value.
  //  double dyn_num_trials_multiplier = 3.0;

  // Number of random trials to estimate model from random subset.
  size_t min_num_trials = 0;
  size_t max_num_trials = std::numeric_limits<size_t>::max();

  void Check() const {
    CHECK_GT(sigmaMax, 0);
    CHECK_GE(min_inlier_ratio, 0);
    CHECK_LE(min_inlier_ratio, 1);
    CHECK_GE(confidenceI, 0);
    CHECK_LE(confidenceI, 1);
    CHECK_GE(confidenceIIB, 0);
    CHECK_LE(confidenceIIB, 1);
    CHECK_GE(confidenceIIT, 0);
    CHECK_LE(confidenceIIT, 1);
    CHECK_LE(min_num_trials, max_num_trials);
  }
};

template <typename Estimator, typename SupportMeasurer = InlierSupportMeasurer,
          typename Sampler = RandomSampler>
class LRTSAC {
 public:
  struct Report {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Whether the estimation was successful.
    bool success = false;

    // The number of RANSAC trials / iterations.
    size_t num_trials = 0;

    // The support of the estimated model.
    typename SupportMeasurer::Support support;

    // Boolean mask which is true if a sample is an inlier.
    std::vector<char> inlier_mask;

    // The estimated model.
    typename Estimator::M_t model;
  };

  explicit LRTSAC(const LRTSACOptions& options);

  // Determine the maximum number of trials required to sample at least one
  // outlier-free random set of samples with the specified confidence,
  // given the inlier ratio.
  //
  // @param eps				        The inlier ratio.
  // @param num_samples				The total number of samples.
  // @param confidenceIIT			Confidence that one sample is
  //								outlier-free.
  // @param confidenceIIB                       Confidence in the early bailout.
  //
  // @return               The required number of iterations.
  static size_t ComputeNumTrials(const double eps, const size_t num_samples,
                                 const double confidenceIIT,
                                 const double confidenceIIB);

  // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y,
                  size_t imagesDimensions[]);

  // Objects used in RANSAC procedure. Access useful to define custom behavior
  // through options or e.g. to compute residuals.
  Estimator estimator;
  Sampler sampler;
  SupportMeasurer support_measurer;

 protected:
  LRTSACOptions options_;

 private:
  double _sigmaMin;  ///< Minimum sigma in the range to try.
  int _B;            ///< Frequency to check for early bailout.
  double _minL;      ///< Min log-likelihood deduced from alpha

  void initSigma(std::vector<double>& Sigma) const;
  double likelihood(double eps, double sigma, size_t imagesDimensions[]) const;
  double bisectLikelihood(double sigma, double L, const size_t num_samples,
                          size_t imagesDimensions[]) const;
  void computeEpsMin(std::vector<double>& Sigma, std::vector<double>& epsMin,
                     double L, const size_t num_samples,
                     size_t imagesDimensions[]) const;
  bool computeEps(const typename Estimator::M_t& model,
                  const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y,
                  const std::vector<double>& Sigma, std::vector<double>& eps,
                  const int num_samples,
                  const std::vector<double>& epsMin) const;

  double bestSigma(const std::vector<double>& Sigma,
                   const std::vector<double>& eps, double& L,
                   double& epsBest, size_t imagesDimensions[]) const;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename SupportMeasurer, typename Sampler>
LRTSAC<Estimator, SupportMeasurer, Sampler>::LRTSAC(
    const LRTSACOptions& options)
    : sampler(Sampler(Estimator::kMinNumSamples)), options_(options) {
  options.Check();
  std::cout << "THIS IS IT";

  _sigmaMin = 0.25;
  _B = 100;  // Should be adjusted to balance bailout test and processing time

  if (options_.sigmaMax < _sigmaMin) _sigmaMin = options_.sigmaMax;
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
size_t LRTSAC<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
    const double eps, const size_t num_samples, const double confidenceIIT,
    const double confidenceIIB) {
  double num = std::log(1 - confidenceIIT);
  double den = std::pow(eps, num_samples);
  den = std::log(1 - confidenceIIB * den);
  if (den == 0)  // Happens if _cpIIB*den<<1
    num = 1;     // So as to return +infty
  return (size_t)(num / den);
}

/// Fill set of sigma values.
/// Geometric progression from _sigmaMin to sigmaMax.
template <typename Estimator, typename SupportMeasurer, typename Sampler>
void LRTSAC<Estimator, SupportMeasurer, Sampler>::initSigma(
    std::vector<double>& Sigma) const {
  const double sigmaMultiplier = sqrt(2.0);
  Sigma.push_back(_sigmaMin);
  while (true) {
    double sigma = Sigma.back() * sigmaMultiplier;
    if (sigma > options_.sigmaMax) break;
    Sigma.push_back(sigma);
  }
  if (Sigma.back() != options_.sigmaMax) Sigma.push_back(options_.sigmaMax);
}

/// Computation of the log-likelihood function. Equation (10)
template <typename Estimator, typename SupportMeasurer, typename Sampler>
double LRTSAC<Estimator, SupportMeasurer, Sampler>::likelihood(
    double eps, double sigma, size_t imagesDimensions[]) const {
  double p = Estimator::pSigma(sigma, imagesDimensions), q = 1 - p;
  if (p < 1.0e-10 || q < 1.0e-10)
    throw std::domain_error(
        "Error likelihood:\n"
        "pSigma too small or too close to 1.");
  if (eps < p) return 0;

  return (eps == 1)
             ? -std::log(p)
             : eps * std::log(eps / p) + (1 - eps) * std::log((1 - eps) / q);
}

/// Bisection based on the likelihood function (in algorithm 3).
/// Find inlier ratio at given \a sigma to reach log-likelihood value \a L.
/// Granularity is 1/NbData.
template <typename Estimator, typename SupportMeasurer, typename Sampler>
double LRTSAC<Estimator, SupportMeasurer, Sampler>::bisectLikelihood(
    double sigma, double L, const size_t num_samples,
    size_t imagesDimensions[]) const {
  double iMin = 0, iMax = 1;
  double LMin = likelihood(iMin, sigma, imagesDimensions);
  double LMax = likelihood(iMax, sigma, imagesDimensions);

  if (L <= LMin) return iMin;
  if (L >= LMax) return iMax;

  while ((iMax - iMin) * num_samples > 1.0) {
    double iMid = (iMin + iMax) * 0.5;
    double LMid = likelihood(iMid, sigma, imagesDimensions);
    assert(LMin <= LMid && LMid <= LMax);
    if (L < LMid) {
      iMax = iMid;
      LMax = LMid;
    } else {
      iMin = iMid;
      LMin = LMid;
    }
  }
  return iMin;
}

/// Compute min epsilon for each sigma to reach log-likelihood \a L.
/// This may also reduce the maximum sigma (algorithm 3).
template <typename Estimator, typename SupportMeasurer, typename Sampler>
void LRTSAC<Estimator, SupportMeasurer, Sampler>::computeEpsMin(
    std::vector<double>& Sigma, std::vector<double>& epsMin, double L,
    const size_t num_samples, size_t imagesDimensions[]) const {
  std::vector<double>::iterator it = Sigma.begin();
  for (int i = 0; it != Sigma.end(); ++it, ++i) {
    if (likelihood(1, *it, imagesDimensions) <= L) break;
    epsMin[i] = (likelihood(0, *it, imagesDimensions) >= L)
                    ? 0
                    : bisectLikelihood(*it, L, num_samples, imagesDimensions);
  }
  if (options_.reduceSigma) Sigma.erase(it, Sigma.end());
}

/// Computation of the inlier ratios (\a eps) for each sigma (algorithm 4).
/// Early bailout may occur if the model is unlikely a better one.
/// \param model Current model to test
/// \param Sigma Set of possible values for sigma
/// \param[out] eps Inlier ratio for each value of sigma
/// \param[out] vpm Number of applied verifications
/// \param epsMin Min eps value for better model (used only for bailout)
/// \return Indicate whether eps is exact (no early bailout)
template <typename Estimator, typename SupportMeasurer, typename Sampler>
bool LRTSAC<Estimator, SupportMeasurer, Sampler>::computeEps(
    const typename Estimator::M_t& model,
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y,
    const std::vector<double>& Sigma, std::vector<double>& eps,
    const int num_samples, const std::vector<double>& epsMin) const {
  const double increment = 1.0 / num_samples;
  for (int j = 0, bailCount = 0; j < num_samples; j++) {
    double error = estimator.Residual(X[j], Y[j], model);

    for (size_t i = 0; i < Sigma.size(); i++)
      if (error <= Sigma[i] * Sigma[i]) eps[i] += increment;

    if (options_.confidenceIIB < 1 && ++bailCount == _B) {
      bailCount = 0;  // Round counter, cheaper than Euclidean division
      double tau = std::sqrt(-(std::log(1 - options_.confidenceIIB) -
                               std::log(std::floor(num_samples / _B))) /
                             (2 * (j + 1)));  // (19)
      bool bailout = true;
      for (size_t i = 0; bailout && i < Sigma.size(); i++)
        if (eps[i] * num_samples >= (j + 1) * (epsMin[i] - tau))
          bailout = false;
      if (bailout) return false;
    }
  }
  return true;
}

/// Find sigma leading to best log-likelihood based on inlier ratios.
/// Algorigthm 2, line 7.
/// \param Sigma Set of possible values for sigma
/// \param eps Inlier ratio for each value of sigma
/// \param[out] L The highest log-likelihood
/// \param[out] bestEps Inlier ratio for best sigma
/// \return The value of the best sigma
template <typename Estimator, typename SupportMeasurer, typename Sampler>
double LRTSAC<Estimator, SupportMeasurer, Sampler>::bestSigma(
    const std::vector<double>& Sigma, const std::vector<double>& eps, double& L,
    double& epsBest, size_t imagesDimensions[]) const {
  double sigma = 0;
  L = -1.0;
  for (size_t i = 0; i < Sigma.size(); i++) {
    double lambda = likelihood(eps[i], Sigma[i], imagesDimensions);
    if (lambda > L) {
      L = lambda;
      sigma = Sigma[i];
      epsBest = eps[i];
    }
  }
  return sigma;
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename LRTSAC<Estimator, SupportMeasurer, Sampler>::Report
LRTSAC<Estimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y,
    size_t imagesDimensions[]) {
  CHECK_EQ(X.size(), Y.size());

  const size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;
  double best_Likelihood = 0;
  double best_Sigma = options_.sigmaMax;

  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  if (options_.confidenceI > 0) {
    // minL computed by inverting chi2 cumulative distribution
    std::pair<int, double> param = {Estimator::nDegreeOfFreedom + 2,
                                    options_.confidenceI};
    if (chi2Table.find(param) == chi2Table.end())
      throw std::invalid_argument("LRTSac's chi2 value not tabulated");
    _minL = chi2Table[param] / (2 * num_samples);  // (13)
  } else
    _minL = 0;

  typename SupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;

  bool abort = false;

  //  const double max_residual = options_.sigmaMax * options_.sigmaMax;

  std::vector<double> residuals(num_samples);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

  sampler.Initialize(num_samples);

  size_t max_num_trials = options_.max_num_trials;
  max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
  size_t dyn_max_num_trials = max_num_trials;

  // Computation of array of values for sigma
  std::vector<double> Sigma;
  initSigma(Sigma);

  std::vector<double> epsMin(Sigma.size(), 0.0);
  if (_minL > 0) {
    if (options_.confidenceIIT < 1 || options_.confidenceIIB < 1 ||
        options_.reduceSigma)
      computeEpsMin(Sigma, epsMin, _minL, num_samples, imagesDimensions);
    if (options_.confidenceIIT < 1)
      dyn_max_num_trials = std::min(
          dyn_max_num_trials,
          ComputeNumTrials(epsMin.front(), num_samples, options_.confidenceIIT,
                           options_.confidenceIIB));
  }

  for (report.num_trials = 0; report.num_trials < max_num_trials;
       ++report.num_trials) {
    if (abort) {
      report.num_trials += 1;
      break;
    }

    sampler.SampleXY(X, Y, &X_rand, &Y_rand);

    // Estimate model for current subset.
    const std::vector<typename Estimator::M_t> sample_models =
        estimator.Estimate(X_rand, Y_rand);

    // Iterate through all estimated models.
    for (const auto& sample_model : sample_models) {
      std::vector<double> eps(Sigma.size(), 0);  // Inlier ratios
      bool noBailout =
          computeEps(sample_model, X, Y, Sigma, eps, num_samples, epsMin);
      if (!noBailout) continue;

      double L, epsBest = 0;
      double sigma = bestSigma(Sigma, eps, L, epsBest, imagesDimensions);

      // Save as best subset if better than all previous subsets.
      if (L > best_Likelihood) {
        best_Likelihood = L;
        best_Sigma = sigma;
        best_model = sample_model;

        if (options_.confidenceIIT < 1 || options_.confidenceIIB < 1 ||
            options_.reduceSigma)
          computeEpsMin(Sigma, epsMin, best_Likelihood, num_samples, imagesDimensions);
        if (options_.confidenceIIT < 1 && !Sigma.empty())
          dyn_max_num_trials = std::min(
              dyn_max_num_trials,
              ComputeNumTrials(epsMin.front(), num_samples,
                               options_.confidenceIIT, options_.confidenceIIB));
      }
      if (report.num_trials >= dyn_max_num_trials &&
          report.num_trials >= options_.min_num_trials) {
        abort = true;
        break;
      }
    }
  }

  report.support = best_support;
  report.model = best_model;

  // No valid model was found.
  if (best_Likelihood < _minL) {
    return report;
  }

  report.success = true;

  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  estimator.Residuals(X, Y, report.model, &residuals);
  CHECK_EQ(residuals.size(), num_samples);

  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals.size(); ++i) {
    if (residuals[i] <= best_Sigma) {
      report.inlier_mask[i] = true;
      report.support.num_inliers += 1;
      report.support.residual_sum += residuals[i];
    }
  }

  return report;
}

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_LRTSAC_H
