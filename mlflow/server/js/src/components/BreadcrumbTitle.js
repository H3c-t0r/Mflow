import React, { Component } from "react";
import PropTypes from "prop-types";
import { Experiment } from "../sdk/MlflowMessages";
import { Link } from 'react-router-dom';
import Routes from "../Routes";
import './BreadcrumbTitle.css';
import Utils from '../utils/Utils';

/**
 * A title component that creates a <h1> with breadcrumbs pointing to an experiment and optionally
 * a run or a run comparison page.
 */
export default class BreadcrumbTitle extends Component {
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(String), // Optional because not all pages are nested under runs
    runNames: PropTypes.arrayOf(String),
    title: PropTypes.any.isRequired,
  };

  static MAX_NAME_DISPLAY_LENGTH = 32;

  render() {
    const {experiment, runUuids, runNames, title} = this.props;
    const experimentId = experiment.getExperimentId();
    const experimentLink = (
      <Link to={Routes.getExperimentPageRoute(experimentId)}>
        {Utils.truncateString(experiment.getName(), BreadcrumbTitle.MAX_NAME_DISPLAY_LENGTH)}
      </Link>
    );
    let runsLink = null;
    if (runUuids) {
      runsLink = (runUuids.length === 1 ?
        <Link to={Routes.getRunPageRoute(experimentId, runUuids[0])} key="link">
          {Utils.truncateString(runNames[0], BreadcrumbTitle.MAX_NAME_DISPLAY_LENGTH)}
        </Link>
        :
        <Link to={Routes.getCompareRunPageRoute(runUuids, experimentId)} key="link">
          Comparing {runUuids.length} Runs
        </Link>
      );
    }
    const chevron = <i className="fas fa-chevron-right breadcrumb-chevron" key="chevron"/>;
    return (
      <h1 className="breadcrumb-header">
        {experimentLink}
        {chevron}
        { runsLink ? [runsLink, chevron] : [] }
        {Utils.truncateString(title, BreadcrumbTitle.MAX_NAME_DISPLAY_LENGTH)}
      </h1>
    );
  }
}

