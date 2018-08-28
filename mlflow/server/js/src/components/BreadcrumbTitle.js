import React, { Component } from "react";
import PropTypes from "prop-types";
import { Experiment } from "../sdk/MlflowMessages";
import { Link } from 'react-router-dom';
import Routes from "../Routes";
import Utils from '../utils/Utils';
import DropdownMenuView from './DropdownMenuView';

/**
 * A title component that creates a <h1> with breadcrumbs pointing to an experiment and optionally
 * a run or a run comparison page.
 */

const DROPDOWN_MENU = 'dropdownMenu';


export default class BreadcrumbTitle extends Component {
  constructor(props) {
    super(props);
    this.onSetRunName = this.onSetRunName.bind(this);
    debugger;
    this.onSetTag = this.props.onSetTag.bind(this);
    this.menuShowing = false;
  }
  // title={Utils.getRunDisplayName(this.props.tags, run.getRunUuid())}
  static propTypes = {
    experiment: PropTypes.instanceOf(Experiment).isRequired,
    runUuids: PropTypes.arrayOf(String), // Optional because not all pages are nested under runs
    // TODO this might need to be an array for multiple runs?
    tags: PropTypes.object,
    onSetTag: PropTypes.func,
  };

  onSetRunName(event) {
    event.preventDefault();
    if (event.target.value) {
      this.props.onSetTag(Utils.getRunTagName(), event.target.value);
    }
  }

  /**
   * Hide the dropdown menu for a table item.
   **/
  hideMenu() {
    this.setState({
      menuShowing: null,
      above: false,
      dropdownHeight: 0,
    });
  }

  renameRun() {
    console.log("Hi! In renameRun() in BreadCrumbTitle.js");
  }

  getMenuItems() {
    // Table specific menu options
    const menuItems = [];

    const renameOnClick = this.renameRun.bind(this);
    menuItems.push(
      <a
        key='rename-item'
        data-name='Rename'
        className='sidebar-dropdown-link'
        onClick={renameOnClick}
      >Rename
      </a>
    );
    return menuItems;
  }

  renderDropdown() {
    return (<DropdownMenuView
            ref={DROPDOWN_MENU}
            getItems={this.getMenuItems}
            outsideClickHandler={this.hideMenu}
            ignoreClickClasses={['sidebar-dropdown']}
          />)
  }

  render() {
    const {experiment, runUuids, title} = this.props;
    const experimentId = experiment.getExperimentId();
    const experimentLink = (
      <Link to={Routes.getExperimentPageRoute(experimentId)}>
        {experiment.getName()}
      </Link>
    );
    let runsLink = null;
    if (runUuids) {
      runsLink = (runUuids.length === 1 ?
        <div>
          <Link to={Routes.getRunPageRoute(experimentId, runUuids[0])} key="link">
            {Utils.getRunDisplayName(this.props.tags, runUuids[0])}
          </Link>
          {this.state.menuShowing ?
            this.renderDropdown(this.state.menuShowing.model, this.state.menuShowing.position,
              this.state.above, this.state.dropdownHeight) :
            null}

        </div>
        :
        <Link to={Routes.getCompareRunPageRoute(runUuids, experimentId)} key="link">
          Comparing {runUuids.length} Runs
        </Link>
      );
    }
    const chevron = <i className="fas fa-chevron-right breadcrumb-chevron" key="chevron"/>;
    return (
      <h1>
        {experimentLink}
        {chevron}
        { runsLink ? [runsLink, chevron] : [] }
        {title}
      </h1>
    );
  }
}
