import React from 'react';
import { connect } from 'react-redux';
import Utils from '../utils/Utils';
import RequestStateWrapper from './RequestStateWrapper';
import { getMetricHistoryApi, getUUID } from '../Actions';
import PropTypes from 'prop-types';
import _ from 'lodash';
import { MetricsPlotView } from './MetricsPlotView';
import { getRunTags } from '../reducers/Reducers';
import { MetricsPlotControls, X_AXIS_RELATIVE, X_AXIS_STEP } from './MetricsPlotControls';
import qs from 'qs';
import { withRouter } from 'react-router-dom';
import Routes from '../Routes';

export const CHART_TYPE_LINE = 'line';
export const CHART_TYPE_BAR = 'bar';

class MetricsPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    metricKey: PropTypes.string.isRequired,
    latestMetricsByRunUuid: PropTypes.object.isRequired,
    distinctMetricKeys: PropTypes.arrayOf(String).isRequired,
    metricsWithRunInfoAndHistory: PropTypes.arrayOf(Object).isRequired,
    getMetricHistoryApi: PropTypes.func.isRequired,
    location: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired,
    runDisplayNames: PropTypes.arrayOf(String).isRequired,
  };

  constructor(props) {
    super(props);
    const plotMetricKeys = MetricsPlotPanel.getPlotMetricKeysFromUrl(props.location.search);
    const selectedMetricKeys = plotMetricKeys.length ? plotMetricKeys : [props.metricKey];
    this.state = {
      selectedXAxis: X_AXIS_RELATIVE,
      selectedMetricKeys,
      showDot: false,
      historyRequestIds: [],
      yAxisLogScale: false,
      lineSmoothness: 0,
    };
    this.loadMetricHistory(this.props.runUuids, selectedMetricKeys);
  }

  static predictChartType(metrics) {
    // Show bar chart when every metric has exactly 1 metric history
    if (
      metrics &&
      metrics.length &&
      _.every(metrics, (metric) => metric.history && metric.history.length === 1)
    ) {
      return CHART_TYPE_BAR;
    }
    return CHART_TYPE_LINE;
  }

  static getPlotMetricKeysFromUrl = (search) => JSON.parse(qs.parse(search)['plot_metric_keys']);

  isComparing = () => {
    const params = qs.parse(this.props.location.search);
    const runs = params && params['?runs'];
    return runs ? JSON.parse(runs).length > 1 : false;
  };

  updateUrlWithSelectedMetrics(selectedMetricKeys) {
    const { runUuids, metricKey, location, history } = this.props;
    const params = qs.parse(location.search);
    const experimentId = params['experiment'];
    history.push(Routes.getMetricPageRoute(runUuids, metricKey, experimentId, selectedMetricKeys));
  }

  loadMetricHistory = (runUuids, metricKeys) => {
    const requestIds = [];
    const { latestMetricsByRunUuid } = this.props;
    runUuids.forEach((runUuid) => {
      metricKeys.forEach((metricKey) => {
        if (latestMetricsByRunUuid[runUuid][metricKey]) {
          const id = getUUID();
          this.props.getMetricHistoryApi(runUuid, metricKey, id);
          requestIds.push(id);
        }
      });
    });
    return requestIds;
  };

  getAllMetricKeys = () => {
    const { distinctMetricKeys } = this.props;
    return distinctMetricKeys.map((metricKey) => ({
      title: metricKey,
      value: metricKey,
      key: metricKey,
    }));
  };

  getMetrics = () => {
    /* eslint-disable no-param-reassign */
    const selectedMetricsSet = new Set(this.state.selectedMetricKeys);
    const { selectedXAxis } = this.state;
    const { metricsWithRunInfoAndHistory } = this.props;

    // Take only selected metrics
    const metrics = metricsWithRunInfoAndHistory.filter((m) => selectedMetricsSet.has(m.metricKey));

    // Sort metric history based on selected x-axis
    metrics.forEach((metric) => {
      const isStep =
        selectedXAxis === X_AXIS_STEP && metric.history[0] && _.isNumber(metric.history[0].step);
      metric.history = _.sortBy(metric.history, isStep ? ['step', 'timestamp'] : 'timestamp');
    });
    return metrics;
  };

  handleYAxisLogScaleChange = (yAxisLogScale) => {
    this.setState({ yAxisLogScale });
  };

  handleXAxisChange = (e) => {
    this.setState({ selectedXAxis: e.target.value });
  };

  handleMetricsSelectChange = (metricValues, metricLabels, { triggerValue }) => {
    const requestIds = this.loadMetricHistory(this.props.runUuids, [triggerValue]);
    this.setState((prevState) => ({
      selectedMetricKeys: metricValues,
      historyRequestIds: [...prevState.historyRequestIds, ...requestIds],
    }));
    this.updateUrlWithSelectedMetrics(metricValues);
  };

  handleShowDotChange = (showDot) => this.setState({ showDot });

  handleLineSmoothChange = (lineSmoothness) => this.setState({ lineSmoothness });

  render() {
    const { runUuids, runDisplayNames } = this.props;
    const {
      historyRequestIds,
      showDot,
      selectedXAxis,
      selectedMetricKeys,
      yAxisLogScale,
      lineSmoothness,
    } = this.state;
    const metrics = this.getMetrics();
    const chartType = MetricsPlotPanel.predictChartType(metrics);
    return (
      <div className='metrics-plot-container'>
        <MetricsPlotControls
          allMetricKeys={this.getAllMetricKeys()}
          selectedXAxis={selectedXAxis}
          selectedMetricKeys={selectedMetricKeys}
          handleXAxisChange={this.handleXAxisChange}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleShowDotChange={this.handleShowDotChange}
          handleYAxisLogScaleChange={this.handleYAxisLogScaleChange}
          handleLineSmoothChange={this.handleLineSmoothChange}
          chartType={chartType}
        />
        <RequestStateWrapper requestIds={historyRequestIds}>
          <MetricsPlotView
            runUuids={runUuids}
            runDisplayNames={runDisplayNames}
            xAxis={selectedXAxis}
            metrics={this.getMetrics()}
            metricKeys={selectedMetricKeys}
            showDot={showDot}
            chartType={chartType}
            isComparing={this.isComparing()}
            yAxisLogScale={yAxisLogScale}
            lineSmoothness={lineSmoothness}
          />
        </RequestStateWrapper>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { runUuids } = ownProps;
  const { latestMetricsByRunUuid, metricsByRunUuid } = state.entities;

  // All metric keys from all runUuids, non-distinct
  const metricKeys = _.flatMap(runUuids, (runUuid) => {
    const latestMetrics = latestMetricsByRunUuid[runUuid];
    return latestMetrics ? Object.keys(latestMetrics) : [];
  });
  const distinctMetricKeys = [...new Set(metricKeys)].sort();

  const runDisplayNames = [];

  // Flat array of all metrics, with history and information of the run it belongs to
  // This is used for underlying MetricsPlotView & predicting chartType for MetricsPlotControls
  const metricsWithRunInfoAndHistory = _.flatMap(runUuids, (runUuid) => {
    const runDisplayName = Utils.getRunDisplayName(getRunTags(runUuid, state), runUuid);
    runDisplayNames.push(runDisplayName);
    const metricsHistory = metricsByRunUuid[runUuid];
    return metricsHistory
      ? Object.keys(metricsHistory).map((metricKey, index) => {
        const history = metricsHistory[metricKey].map((entry) => ({
          key: entry.key,
          value: entry.value,
          step: Number.parseInt(entry.step, 10) || 0, // default step to 0
          timestamp: Number.parseFloat(entry.timestamp),
        }));
        return { metricKey, history, runUuid, runDisplayName, index };
      })
      : [];
  });

  return {
    runDisplayNames,
    latestMetricsByRunUuid,
    distinctMetricKeys,
    metricsWithRunInfoAndHistory,
  };
};

const mapDispatchToProps = { getMetricHistoryApi };

export default withRouter(
  connect(
    mapStateToProps,
    mapDispatchToProps,
  )(MetricsPlotPanel),
);
