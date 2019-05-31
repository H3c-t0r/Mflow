class Routes {
  static rootRoute = "/";

  static getExperimentPageRoute(experimentId) {
    return `/experiments/${experimentId}`;
  }

  static experimentPageRoute = "/experiments/:experimentId";

  static getRunPageRoute(experimentId, runUuid) {
    return `/experiments/${experimentId}/runs/${runUuid}`;
  }

  static runPageRoute = "/experiments/:experimentId/runs/:runUuid";

  static getMetricPageRoute(runUuids, metricKey, experimentId, plotMetricKeys) {
    return `/metric/${metricKey}?runs=${JSON.stringify(runUuids)}&experiment=${experimentId}` +
      `&plot_metric_keys=${JSON.stringify(plotMetricKeys || [metricKey])}`;
  }

  static metricPageRoute = "/metric/:metricKey";

  static getCompareRunPageRoute(runUuids, experimentId) {
    return `/compare-runs?runs=${JSON.stringify(runUuids)}&experiment=${experimentId}`;
  }

  static compareRunPageRoute = "/compare-runs"

  static getRunSearchPageRoute(paramKeyFilterInput, metricKeyFilterInput, searchInput, lifecycleFilterInput) {
    return `/s?params=${paramKeyFilterInput}&metrics=${metricKeyFilterInput}`
            +`&searchInput=${searchInput}&lifecycle=${lifecycleFilterInput}`
  }

}

export default Routes;
