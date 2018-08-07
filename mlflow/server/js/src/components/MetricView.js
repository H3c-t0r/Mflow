import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { LineChart, BarChart, Bar, XAxis, Tooltip, CartesianGrid, Line, YAxis, ResponsiveContainer, Legend } from 'recharts';
import { connect } from 'react-redux';
import Utils from '../utils/Utils';
import { getMetricsByKey } from '../reducers/MetricReducer';
import './MetricView.css';
import {Experiment} from "../sdk/MlflowMessages";
import {getExperiment} from "../reducers/Reducers";
import Routes from "../Routes";
import {Link} from "react-router-dom";

const COLORS = [
  "#993955",
  "#AE76A6",
  "#A3C3D9",
  "#364958",
  "#FF82A9",
  "#FFC0BE",
];

class MetricView extends Component {
  static propTypes = {
    // Experiment is optional but we'll link back to an experiment if we have it
    experiment: PropTypes.instanceOf(Experiment),
    title: PropTypes.element.isRequired,
    // Object with keys from Metric json and also
    metrics: PropTypes.arrayOf(Object).isRequired,
    runUuids: PropTypes.arrayOf(String).isRequired,
  };


  getTitle() {
    const experiment = this.props.experiment;
    const runUuids = this.props.runUuids;
    if (experiment) {
      const experimentId = experiment.getExperimentId();
      const experimentLink = (
        <Link to={Routes.getExperimentPageRoute(experimentId)}>
          {experiment.getName()}
        </Link>
      );
      const runsLink = (runUuids.length === 1 ?
          <Link to={Routes.getRunPageRoute(experimentId, runUuids[0])}>
            Run {runUuids[0]}
          </Link>
        :
          <Link to={Routes.getCompareRunPageRoute(runUuids, experimentId)}>
            Comparing {runUuids.length} Runs
          </Link>
      );
      return (
        <h1 className="MetricView-title">
          {experimentLink}
          <i className="fas fa-chevron-right breadcrumb-chevron"></i>
          {runsLink}
          <i className="fas fa-chevron-right breadcrumb-chevron"></i>
          {this.props.title}
        </h1>
      )
    } else {
      // Without an experiment ID, we can't link back to the run and compare pages;
      // this might come up if we allow cross-run comparison and we'll have to design some
      // other mechanism to go back in that case.
      return (
        <h1 className="MetricView-title">
          {this.props.title}
        </h1>
      )
    }
  }

  render() {
      if (this.props.metrics.length === 1) {
          return (
              <div className="MetricView">
                  {this.getTitle()}
                  <ResponsiveContainer width="100%" aspect={1.55}>
                      <BarChart
                          data={this.props.metrics}
                          margin={{top: 10, right: 10, left: 10, bottom: 10}}
                      >
                          <XAxis dataKey="index"/>
                          <Tooltip isAnimationActive={false} labelStyle={{display: "none"}}/>
                          <CartesianGrid strokeDasharray="3 3"/>
                          <Legend verticalAlign="bottom"/>
                          <YAxis/>
                          {this.props.runUuids.map((uuid, idx) => (
                              <Bar dataKey={uuid}
                                   key={uuid}
                                   isAnimationActive={false}
                                   fill={COLORS[idx % COLORS.length]}/>
                          ))}
                      </BarChart>
                  </ResponsiveContainer>
              </div>
          )
      } else {
          return (
              <div className="MetricView">
                  {this.getTitle()}
                  <ResponsiveContainer width="100%" aspect={1.55}>
                      <LineChart
                          data={Utils.convertTimestampToInt(this.props.metrics)}
                          margin={{top: 10, right: 10, left: 10, bottom: 10}}
                      >
                          <XAxis dataKey="index" type="number"/>
                          <Tooltip isAnimationActive={false} labelStyle={{display: "none"}}/>
                          <CartesianGrid strokeDasharray="3 3"/>
                          <Legend verticalAlign="bottom"/>
                          <YAxis/>
                          {this.props.runUuids.map((uuid, idx) => (
                              <Line type="linear"
                                    dataKey={uuid}
                                    key={uuid}
                                    isAnimationActive={false}
                                    connectNulls
                                    stroke={COLORS[idx % COLORS.length]}/>
                          ))}
                      </LineChart>
                  </ResponsiveContainer>
              </div>
          )
      }
  }
}

const mapStateToProps = (state, ownProps) => {
  const { metricKey, runUuids, experimentId } = ownProps;
  const experiment = experimentId !== null ? getExperiment(experimentId, state) : null;
  let maxLength = 0;
  runUuids.forEach(runUuid => {
    maxLength = Math.max(maxLength, getMetricsByKey(runUuid, metricKey, state).length)
  });
  const metrics = new Array(maxLength);
  for (let i = 0; i < metrics.length; i++) {
    metrics[i] = {index: i};
  }
  runUuids.forEach(runUuid => {
    const entries = getMetricsByKey(runUuid, metricKey, state);
    for (let i = 0; i < entries.length; i++) {
      metrics[i][runUuid] = entries[i].value;
    }
  });
  return {
    experiment,
    metrics,
    title: <span>{metricKey}</span>,
    runUuids: runUuids,
  }
};

export default connect(mapStateToProps)(MetricView);
