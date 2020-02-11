import React from 'react';
import { connect } from 'react-redux';
import Utils from '../utils/Utils';
import RequestStateWrapper from './RequestStateWrapper';
import { getMetricHistoryApi, getUUID } from '../Actions';
import PropTypes from 'prop-types';
import _ from 'lodash';
import { MetricsPlotView } from './MetricsPlotView';
import { getRunTags } from '../reducers/Reducers';
import {
  MetricsPlotControls,
  X_AXIS_RELATIVE,
  X_AXIS_STEP,
  X_AXIS_WALL,
} from './MetricsPlotControls';
import qs from 'qs';
import { withRouter } from 'react-router-dom';
import Routes from '../Routes';

export const CHART_TYPE_LINE = 'line';
export const CHART_TYPE_BAR = 'bar';

export class MetricsPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    metricKey: PropTypes.string.isRequired,
    // A map of { runUuid : { metricKey: value } }
    latestMetricsByRunUuid: PropTypes.object.isRequired,
    // An array of distinct metric keys across all runUuids
    distinctMetricKeys: PropTypes.arrayOf(String).isRequired,
    // An array of { metricKey, history, runUuid, runDisplayName }
    metricsWithRunInfoAndHistory: PropTypes.arrayOf(Object).isRequired,
    getMetricHistoryApi: PropTypes.func.isRequired,
    location: PropTypes.object.isRequired,
    history: PropTypes.object.isRequired,
    runDisplayNames: PropTypes.arrayOf(String).isRequired,
  };

  constructor(props) {
    super(props);
    const plotMetricKeys = Utils.getPlotMetricKeysFromUrl(props.location.search);
    const selectedMetricKeys = plotMetricKeys.length ? plotMetricKeys : [props.metricKey];
    this.state = {
      selectedXAxis: X_AXIS_RELATIVE,
      selectedMetricKeys,
      showPoint: false,
      historyRequestIds: [],
      lineSmoothness: 0,
    };
    this.loadMetricHistory(this.props.runUuids, this.state.selectedMetricKeys);
  }

  updateUrlWithSelectedMetrics(selectedMetricKeys) {
    const { runUuids, metricKey, location, history } = this.props;
    const params = qs.parse(location.search);
    const experimentId = params['experiment'];
    history.push(Routes.getMetricPageRoute(runUuids, metricKey, experimentId, selectedMetricKeys));
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

  static isComparing(search) {
    const params = qs.parse(search);
    const runs = params && params['?runs'];
    return runs ? JSON.parse(runs).length > 1 : false;
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
      // Metric history can be large. Doing an in-place here to save memory
      metric.history.sort(isStep ? Utils.compareByStepAndTimestamp : Utils.compareByTimestamp);
    });
    return metrics;
  };

  handleYAxisLogScaleChange = (yAxisLogScale) => {
    const state = this.state;
    const newLayout = _.cloneDeep(state.layout);
    // If yaxis was already explicitly specified, convert range to appropriate coordinates
    // for log axis (base 10), and vice versa. When converting to log scale, handle negative values
    // by deferring to plotly autorange
    const newAxisType = yAxisLogScale ? "log" : "linear";

    // If plot previously had no y axis range configured, simply set the axis type to log or
    // linear scale appropriately
    if (!state.layout.yaxis || !this.state.layout.yaxis.range) {
      newLayout.yaxis = {type: newAxisType, autorange: true};
      this.setState({layout: newLayout});
      return;
    }

    // At this point, we know the plot previously had a y axis specified with range
    // Convert the range to/from log scale as appropriate
    const oldYRange = state.layout.yaxis.range;
    if (state.layout.yaxis.type === 'log') {
      // We at this point know that there was an old y scale, so should convert it to/from
      // log scale.
      // When converting from log scale to linear scale, only apply conversion if autorange
      // was not true (otherwise restore old axis values)
      if (state.layout.yaxis.autorange) {
        newLayout.yaxis = {
          type: 'linear',
          range: oldYRange,
        };
      } else {
        newLayout.yaxis = {
          type: 'linear',
          range: [Math.pow(10, oldYRange[0]), Math.pow(10, oldYRange[1])],
        };
      }
    } else {
      if (oldYRange[0] < 0) {
        // When converting to log scale, handle negative values as follows:
        // If bottom of old Y range is negative, then simply autorange the plot, so that we
        // can convert back. Otherwise, do the conversion
        newLayout.yaxis = {
          type: 'log',
          range: oldYRange,
          autorange: true,
        };
      } else {
        newLayout.yaxis = {
          type: 'log',
          range: [Math.log(oldYRange[0]) / Math.log(10), Math.log(oldYRange[1]) / Math.log(10)],
        };
      }
    }
    this.setState({layout: newLayout});
  };

  handleXAxisChange = (e) => {
    // Set axis value type, & reset axis scaling via autorange
    const axisType = e.target.value === X_AXIS_WALL ? "date" : "linear";
    const newLayout = {
      ...this.state.layout,
      xaxis: {
        autorange: true,
        type: axisType,
      },
    };
    this.setState({ selectedXAxis: e.target.value, layout: newLayout });
  };

  handleLayoutChange = (newLayout) => {
    // Unfortunately, we need to parse out the x & y axis range changes from the onLayout event...
    // see https://plot.ly/javascript/plotlyjs-events/#update-data
    const newXRange0 = newLayout["xaxis.range[0]"];
    const newXRange1 = newLayout["xaxis.range[1]"];
    const newYRange0 = newLayout["yaxis.range[0]"];
    const newYRange1 = newLayout["yaxis.range[1]"];
    let mergedLayout = {
      ...this.state.layout,
    };
    if (newXRange0) {
      mergedLayout = {
        ...mergedLayout,
        xaxis: {
          range: [newXRange0, newXRange1],
        },
      };
    }
    if (newYRange0) {
      mergedLayout = {
        ...mergedLayout,
        yaxis: {
          range: [newYRange0, newYRange1],
        },
      };
    }
    if (newLayout["xaxis.autorange"]) {
      mergedLayout = {
        ...mergedLayout,
        xaxis: {autorange: true},
      };
    }
    if (newLayout["yaxis.autorange"]) {
      const axisType = this.state.layout && this.state.layout.yaxis &&
        this.state.layout.yaxis.type === 'log' ? "log" : "linear";
      mergedLayout = {
        ...mergedLayout,
        yaxis: {autorange: true, type: axisType},
      };
    }
    this.setState({layout: mergedLayout});
  };


  handleMetricsSelectChange = (metricValues, metricLabels, { triggerValue }) => {
    const requestIds = this.loadMetricHistory(this.props.runUuids, [triggerValue]);
    this.setState((prevState) => ({
      selectedMetricKeys: metricValues,
      historyRequestIds: [...prevState.historyRequestIds, ...requestIds],
    }));
    this.updateUrlWithSelectedMetrics(metricValues);
  };

  handleShowPointChange = (showPoint) => this.setState({ showPoint });

  handleLineSmoothChange = (lineSmoothness) => {
    this.setState({ lineSmoothness });
  };

  render() {
    const { runUuids, runDisplayNames, distinctMetricKeys, location } = this.props;
    const {
      historyRequestIds,
      showPoint,
      selectedXAxis,
      selectedMetricKeys,
      lineSmoothness,
    } = this.state;
    const metrics = this.getMetrics();
    const chartType = MetricsPlotPanel.predictChartType(metrics);
    return (
      <div className='metrics-plot-container'>
        <MetricsPlotControls
          distinctMetricKeys={distinctMetricKeys}
          selectedXAxis={selectedXAxis}
          selectedMetricKeys={selectedMetricKeys}
          handleXAxisChange={this.handleXAxisChange}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleShowPointChange={this.handleShowPointChange}
          handleYAxisLogScaleChange={this.handleYAxisLogScaleChange}
          handleLineSmoothChange={this.handleLineSmoothChange}
          chartType={chartType}
          initialLineSmoothness={lineSmoothness}
          showPoint={showPoint}
        />
        <RequestStateWrapper
            requestIds={historyRequestIds}
            // In this case where there are no history request IDs (e.g. on the
            // initial page load / before we try to load additional metrics),
            // optimistically render the children
            shouldOptimisticallyRender={historyRequestIds.length === 0}
        >
          <MetricsPlotView
            runUuids={runUuids}
            runDisplayNames={runDisplayNames}
            xAxis={selectedXAxis}
            metrics={this.getMetrics()}
            metricKeys={selectedMetricKeys}
            showPoint={showPoint}
            chartType={chartType}
            isComparing={MetricsPlotPanel.isComparing(location.search)}
            lineSmoothness={lineSmoothness}
            extraLayout={this.state.layout}
            onLayoutChange={this.handleLayoutChange}
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
      ? Object.keys(metricsHistory).map((metricKey) => {
        const history = metricsHistory[metricKey].map((entry) => ({
          key: entry.key,
          value: entry.value,
          step: Number.parseInt(entry.step, 10) || 0, // default step to 0
          timestamp: Number.parseFloat(entry.timestamp),
        }));
        return { metricKey, history, runUuid, runDisplayName };
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
