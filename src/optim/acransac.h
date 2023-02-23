// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_OPTIM_ACRANSAC_H_
#define COLMAP_SRC_OPTIM_ACRANSAC_H_

#include <cfloat>
#include <random>
#include <stdexcept>
#include <vector>

#include "optim/random_sampler.h"
#include "optim/ransac.h"
#include "optim/support_measurement.h"
#include "util/alignment.h"
#include "util/logging.h"

namespace colmap {

struct ACRANSACOptions {
  // Maximum error for a sample to be considered as an inlier. Note that
  // the residual of an estimator corresponds to a squared error.
  double sigmaMax = 0.0;

  // A priori assumed minimum inlier ratio, which determines the maximum number
  // of iterations. Only applies if smaller than `max_num_trials`.
  double min_inlier_ratio = 0.1;

  // Number of random trials to estimate model from random subset.
  size_t min_num_trials = 0;
  size_t max_num_trials = std::numeric_limits<size_t>::max();

  void Check() const {
    CHECK_GE(sigmaMax, 0);
    CHECK_GE(min_inlier_ratio, 0);
    CHECK_LE(min_inlier_ratio, 1);
    CHECK_LE(min_num_trials, max_num_trials);
  }
};

template <typename Estimator, typename SupportMeasurer = InlierSupportMeasurer,
          typename Sampler = RandomSampler>
class ACRANSAC {
 public:
  typedef typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report Report;

  explicit ACRANSAC(const ACRANSACOptions& options);

  // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
  //
  // @param X              Independent variables.
  // @param Y              Dependent variables.
  //
  // @return               The report with the results of the estimation.
  Report Estimate(const std::vector<typename Estimator::X_t>& X,
                  const std::vector<typename Estimator::Y_t>& Y,
                  size_t imagesDimensions[], const double scalingFactor);

  // Objects used in RANSAC procedure. Access useful to define custom behavior
  // through options or e.g. to compute residuals.
  Estimator estimator;
  Sampler sampler;
  SupportMeasurer support_measurer;

 protected:
  ACRANSACOptions options_;
  /// Probabilities of having an error of 1 pixel in left or right image
  double _alpha0Left, _alpha0Right;
  double logalpha0_[2];  ///< Log probability of error<=1, set by subclass

  /// Distance and associated index
  struct ErrorIndex {
    double error;  ///< Square error
    int index;     ///< Correspondence index
    int side;      ///< Error in image 1 (side=0) or 2 (side=1)?
    /// side is not implemented but as we only use ACRANSAC on PNP it is not
    /// needed.
    /// Constructor
    ErrorIndex(double e = 0, int i = 0, int s = 0)
        : error(e), index(i), side(s) {}

    bool operator<(const ErrorIndex& e) const { return (error < e.error); }
  };

  ErrorIndex bestNFA(const std::vector<ErrorIndex>& e, double loge0,
                     double maxThreshold, const std::vector<float>& logc_n,
                     const std::vector<float>& logc_k) const;

  /// logarithm (base 10) of binomial coefficient
  static float logcombi(int k, int n);

  /// tabulate logcombi(.,n)
  void makelogcombi_n(int n, std::vector<float>& l);

  /// tabulate logcombi(k,.)
  void makelogcombi_k(int k, int nmax, std::vector<float>& l);
  /// Refine until convergence removed
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename SupportMeasurer, typename Sampler>
ACRANSAC<Estimator, SupportMeasurer, Sampler>::ACRANSAC(
    const ACRANSACOptions& options)
    : sampler(Sampler(Estimator::kMinNumSamples)), options_(options) {
  options.Check();

  // Determine max_num_trials based on assumed `min_inlier_ratio`.
  const size_t kNumSamples = 100000;
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename ACRANSAC<Estimator, SupportMeasurer, Sampler>::ErrorIndex
ACRANSAC<Estimator, SupportMeasurer, Sampler>::bestNFA(
    const std::vector<ErrorIndex>& e, double loge0, double maxThreshold,
    const std::vector<float>& logc_n, const std::vector<float>& logc_k) const {
  const int startIndex = Estimator::kMinNumSamples;
  const double multError = (Estimator::DistToPoint ? 1.0 : 0.5);

  ErrorIndex bestIndex(std::numeric_limits<double>::infinity(), startIndex, 0);

  const int n = static_cast<int>(e.size());
  for (int k = startIndex + 1; k <= n && e[k - 1].error <= maxThreshold; ++k) {
    double logalpha = logalpha0_[e[k - 1].side] +
                      multError * log10(e[k - 1].error +
                                        std::numeric_limits<double>::epsilon());

    ErrorIndex index(
        loge0 + logalpha * (double)(k - startIndex) + logc_n[k] + logc_k[k], k,
        e[k - 1].side);
    if (index.error < bestIndex.error) bestIndex = index;
  }
  return bestIndex;
}

/// logarithm (base 10) of binomial coefficient
template <typename Estimator, typename SupportMeasurer, typename Sampler>
float ACRANSAC<Estimator, SupportMeasurer, Sampler>::logcombi(int k, int n) {
  if (k >= n || k <= 0) return (0.0);
  if (n - k < k) k = n - k;
  double r = 0.0;
  for (int i = 1; i <= k; i++)
    r += log10((double)(n - i + 1)) - log10((double)i);

  return static_cast<float>(r);
}

/// tabulate logcombi(.,n)
template <typename Estimator, typename SupportMeasurer, typename Sampler>
void ACRANSAC<Estimator, SupportMeasurer, Sampler>::makelogcombi_n(
    int n, std::vector<float>& l) {
  l.resize(n + 1);
  for (int k = 0; k <= n; k++) l[k] = logcombi(k, n);
}

/// tabulate logcombi(k,.)
template <typename Estimator, typename SupportMeasurer, typename Sampler>
void ACRANSAC<Estimator, SupportMeasurer, Sampler>::makelogcombi_k(
    int k, int nmax, std::vector<float>& l) {
  l.resize(nmax + 1);
  for (int n = 0; n <= nmax; n++) l[n] = logcombi(k, n);
}

template <typename Estimator, typename SupportMeasurer, typename Sampler>
typename ACRANSAC<Estimator, SupportMeasurer, Sampler>::Report
ACRANSAC<Estimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y, size_t imagesDimensions[],
    const double scalingFactor) {
  Timer timerLOR;
  timerLOR.Start();
  CHECK_EQ(X.size(), Y.size());


  _alpha0Right = Estimator::pSigma(1, imagesDimensions, false);
  logalpha0_[1] = log10(_alpha0Right);
  _alpha0Left = Estimator::pSigma(1, imagesDimensions, true);
  logalpha0_[0] = log10(_alpha0Left);

  const size_t num_samples = X.size();

  Report report;
  report.success = false;
  report.num_trials = 0;

  if (num_samples < Estimator::kMinNumSamples) {
    return report;
  }

  const double maxThreshold = (options_.sigmaMax > 0)
                                  ? options_.sigmaMax * options_.sigmaMax
                                  :  // Square max error
                                  std::numeric_limits<double>::infinity();

  typename SupportMeasurer::Support best_support;
  typename Estimator::M_t best_model;

  bool abort = false;

  std::vector<ErrorIndex> indexedErrors(num_samples);

  std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
  std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

  sampler.Initialize(num_samples);

  // Possible sampling indices (could change in the optimization phase)
  std::vector<int> vInliers;
  std::vector<typename Estimator::X_t> Xselected(num_samples);
  std::vector<typename Estimator::Y_t> Yselected(num_samples);
  for (int i = 0; i < num_samples; ++i) {
    Xselected[i] = X[i];
    Yselected[i] = Y[i];
  }

  // Precompute log combi
  double loge0 = log10((double)Estimator::NbModels *
                       (num_samples - Estimator::kMinNumSamples));
  std::vector<float> vLogc_n, vLogc_k;
  makelogcombi_n(num_samples, vLogc_n);
  makelogcombi_k(Estimator::kMinNumSamples, num_samples, vLogc_k);

  size_t max_num_trials = options_.max_num_trials;
  max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
  size_t dyn_max_num_trials = max_num_trials;
  size_t num_trials_reserve = max_num_trials / 10;
  dyn_max_num_trials -= num_trials_reserve;

  // Output parameters
  double minNFA = std::numeric_limits<double>::infinity();
  double errorMax = 0;
  int side = 0;

  for (report.num_trials = 0; report.num_trials < max_num_trials;
       ++report.num_trials) {
    if (abort) {
      report.num_trials += 1;
      break;
    }

    sampler.SampleXY(Xselected, Yselected, &X_rand, &Y_rand);
    bool better = false;
    // Estimate model for current subset.
    const std::vector<typename Estimator::M_t> sample_models =
        estimator.Estimate(X_rand, Y_rand);

    // Iterate through all estimated models.
    for (const auto& sample_model : sample_models) {
      for (int i = 0; i < num_samples; ++i) {
        double error = estimator.Residual(X[i], Y[i], sample_model) *
                       scalingFactor * scalingFactor;
        indexedErrors[i] = ErrorIndex(error, i);
      }
      CHECK_EQ(indexedErrors.size(), num_samples);

      // Most meaningful discrimination inliers/outliers
      std::sort(indexedErrors.begin(), indexedErrors.end());
      ErrorIndex best =
          bestNFA(indexedErrors, loge0, maxThreshold, vLogc_n, vLogc_k);

      // Save as best subset if better than all previous subsets.
      if (best.error < minNFA) {  // A better model was found
        if (best.error < 0) best_model = sample_model;
        better = true;
        minNFA = best.error;
        side = best.side;
        vInliers.resize(best.index);
        for (int i = 0; i < best.index; ++i)
          vInliers[i] = indexedErrors[i].index;
        errorMax = indexedErrors[best.index - 1].error;  // Error threshold
        best_model = sample_model;
      }
      // ORSA optimization: draw samples among best set of inliers so far
      if ((better && minNFA < 0) ||
          (report.num_trials + 1 == dyn_max_num_trials && num_trials_reserve)) {
        if (vInliers.empty()) {  // No model found at all so far
          dyn_max_num_trials++;  // Continue to look for any model, even not
                                 // meaningful
          num_trials_reserve--;
        } else {
          std::vector<int>::const_iterator itInlier = vInliers.begin();
          Xselected.resize(vInliers.size());
          Yselected.resize(vInliers.size());
          for (int i = 0; itInlier != vInliers.end(); itInlier++, i++) {
            Xselected[i] = X[*itInlier];
            Yselected[i] = Y[*itInlier];
          }
          if (num_trials_reserve) {
            dyn_max_num_trials = report.num_trials + 1 + num_trials_reserve;
            num_trials_reserve = 0;
          }
        }
      }
      if (report.num_trials >= dyn_max_num_trials &&
          report.num_trials >= options_.min_num_trials) {
        abort = true;
        break;
      }
    }
  }

  if (minNFA >= 0) vInliers.clear();

  report.model = best_model;

  // No valid model was found.
  if (minNFA >= 0) {
    return report;
  }

  report.success = true;
  // Determine inlier mask. Note that this calculates the residuals for the
  // best model twice, but saves to copy and fill the inlier mask for each
  // evaluated model. Some benchmarking revealed that this approach is faster.

  std::vector<double> residuals(num_samples);
  estimator.Residuals(X, Y, report.model, &residuals);
  CHECK_EQ(residuals.size(), num_samples);

  report.inlier_mask.resize(num_samples);
  for (size_t i = 0; i < residuals.size(); ++i) {
    if (residuals[i] * scalingFactor * scalingFactor <= errorMax) {
      report.inlier_mask[i] = true;
      report.support.num_inliers += 1;
      report.support.residual_sum += residuals[i];
    }
  }
  std::cout << "AC-RANSAC final threshold: " << sqrt(errorMax) << std::endl;
  report.ransacTimer = timerLOR.ElapsedSeconds();

  return report;
}

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_ACRANSAC_H_
