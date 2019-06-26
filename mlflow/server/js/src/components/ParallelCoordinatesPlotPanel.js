import React from 'react';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import ParallelCoordinatesPlotView from './ParallelCoordinatesPlotView';
import { ParallelCoordinatesPlotControls } from './ParallelCoordinatesPlotControls';
import { getSharedMetricKeysByRunUuids, getSharedParamKeysByRunUuids } from '../reducers/Reducers';
import _ from 'lodash';

import './ParallelCoordinatesPlotPanel.css';

export class ParallelCoordinatesPlotPanel extends React.Component {
  static propTypes = {
    runUuids: PropTypes.arrayOf(String).isRequired,
    sharedParamKeys: PropTypes.arrayOf(String).isRequired,
    sharedMetricKeys: PropTypes.arrayOf(String).isRequired,
  };

  state = {
    selectedParamKeys: this.props.sharedParamKeys, // default select all parameters
    selectedMetricKeys: this.props.sharedMetricKeys.slice(0, 1),
  };

  handleParamsSelectChange = (paramValues) => {
    this.setState({ selectedParamKeys: paramValues });
  };

  handleMetricsSelectChange = (metricValues) => {
    this.setState({ selectedMetricKeys: metricValues });
  };

  render() {
    const { runUuids, sharedParamKeys, sharedMetricKeys } = this.props;
    const { selectedParamKeys, selectedMetricKeys } = this.state;
    return (
      <div className='parallel-coorinates-plot-panel'>
        <ParallelCoordinatesPlotControls
          paramKeys={sharedParamKeys}
          metricKeys={sharedMetricKeys}
          selectedParamKeys={selectedParamKeys}
          selectedMetricKeys={selectedMetricKeys}
          handleMetricsSelectChange={this.handleMetricsSelectChange}
          handleParamsSelectChange={this.handleParamsSelectChange}
        />
        {(!_.isEmpty(selectedParamKeys) || !_.isEmpty(selectedMetricKeys)) ? (
          <ParallelCoordinatesPlotView
            runUuids={runUuids}
            paramKeys={selectedParamKeys}
            metricKeys={selectedMetricKeys}
          />
        ) : null}
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  const { runUuids } = ownProps;
  const sharedParamKeys = getSharedParamKeysByRunUuids(runUuids, state);
  const sharedMetricKeys = getSharedMetricKeysByRunUuids(runUuids, state);
  return { sharedParamKeys, sharedMetricKeys };
};

export default connect(mapStateToProps)(ParallelCoordinatesPlotPanel);
