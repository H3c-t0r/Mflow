import React, { Component } from 'react';
import PropTypes from 'prop-types';
import { connect } from 'react-redux';
import _ from 'lodash';
import { withRouter } from 'react-router-dom';

import './ExperimentPage.css';
import { getExperimentApi, searchRunsApi, loadMoreRunsApi, searchRunsPayload } from '../actions';
import { searchModelVersionsApi } from '../../model-registry/actions';
import ExperimentView from './ExperimentView';
import RequestStateWrapper from '../../common/components/RequestStateWrapper';
import KeyFilter from '../utils/KeyFilter';
import { ViewType } from '../sdk/MlflowEnums';
import { ExperimentPagePersistedState } from '../sdk/MlflowLocalStorageMessages';
import Utils from '../../common/utils/Utils';
import { ErrorCodes } from '../../common/constants';
import { PermissionDeniedView } from './PermissionDeniedView';
import { Spinner } from '../../common/components/Spinner';
import { getUUID } from '../../common/utils/ActionUtils';
import { MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER } from '../../model-registry/constants';
import { getExperiment } from '../reducers/Reducers';
import { Experiment } from '../sdk/MlflowMessages';
import { injectIntl } from 'react-intl';
import {
  LIFECYCLE_FILTER,
  MODEL_VERSION_FILTER,
  PAGINATION_DEFAULT_STATE,
  MAX_DETECT_NEW_RUNS_RESULTS,
  DETECT_NEW_RUNS_INTERVAL,
  DEFAULT_ORDER_BY_KEY,
  DEFAULT_ORDER_BY_ASC,
  DEFAULT_START_TIME,
  DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  ATTRIBUTE_COLUMN_SORT_KEY,
} from '../constants';

export const isNewRun = (lastRunsRefreshTime, run) => {
  if (run && run.info) {
    const { start_time, end_time } = run.info;

    return start_time >= lastRunsRefreshTime || (end_time !== 0 && end_time >= lastRunsRefreshTime);
  }

  return false;
};

export class ExperimentPage extends Component {
  static propTypes = {
    experimentId: PropTypes.string.isRequired,
    experiment: PropTypes.instanceOf(Experiment),
    getExperimentApi: PropTypes.func.isRequired,
    searchRunsApi: PropTypes.func.isRequired,
    searchModelVersionsApi: PropTypes.func.isRequired,
    loadMoreRunsApi: PropTypes.func.isRequired,
    history: PropTypes.object.isRequired,
    location: PropTypes.object,
    searchForNewRuns: PropTypes.func,
    intl: PropTypes.shape({ formatMessage: PropTypes.func.isRequired }).isRequired,
  };

  static defaultProps = {
    /*
      The runs table reads directly from the redux store, so we are intentionally not using a redux
      action to search for new runs. We do not want to change the runs displayed on the runs table
      when searching for new runs.
     */
    searchForNewRuns: searchRunsPayload,
  };

  constructor(props) {
    super(props);
    const urlState = Utils.getSearchParamsFromUrl(props.location.search);
    this.state = {
      lastRunsRefreshTime: Date.now(),
      numberOfNewRuns: 0,
      // Last experiment, if any, displayed by this instance of ExperimentPage
      lastExperimentId: undefined,
      // Lifecycle filter of runs to display
      lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
      // Filter of model versions to display
      modelVersionFilter: MODEL_VERSION_FILTER.ALL_RUNS,
      ...PAGINATION_DEFAULT_STATE,
      persistedState: {
        paramKeyFilterString: urlState.params === undefined ? '' : urlState.params,
        metricKeyFilterString: urlState.metrics === undefined ? '' : urlState.metrics,
        searchInput: urlState.search === undefined ? '' : urlState.search,
        orderByKey: urlState.orderByKey === undefined ? DEFAULT_ORDER_BY_KEY : urlState.orderByKey,
        orderByAsc:
          urlState.orderByAsc === undefined ? DEFAULT_ORDER_BY_ASC : urlState.orderByAsc === 'true',
        startTime: urlState.startTime === undefined ? DEFAULT_START_TIME : urlState.startTime,
        categorizedUncheckedKeys:
          urlState.categorizedUncheckedKeys === undefined
            ? DEFAULT_CATEGORIZED_UNCHECKED_KEYS
            : urlState.categorizedUncheckedKeys,
      },
    };
  }

  componentDidMount() {
    this.loadData();
    this.detectNewRunsTimer = setInterval(() => this.detectNewRuns(), DETECT_NEW_RUNS_INTERVAL);
  }

  componentDidUpdate(prevProps) {
    this.maybeReloadData(prevProps);
  }

  static getDerivedStateFromProps(props, state) {
    if (props.experimentId !== state.lastExperimentId) {
      return {
        persistedState:
          state.lastExperimentId === undefined
            ? state.persistedState
            : new ExperimentPagePersistedState().toJSON(),
        lastExperimentId: props.experimentId,
        lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
        nextPageToken: null,
      };
    }
    return null;
  }

  componentWillUnmount() {
    clearInterval(this.detectNewRunsTimer);
    this.detectNewRunsTimer = null;
  }

  getExperimentRequestId = getUUID();
  searchRunsRequestId = getUUID();
  searchModelVersionsRequestId = getUUID();
  loadMoreRunsRequestId = getUUID();

  loadData() {
    this.props.getExperimentApi(this.props.experimentId, this.getExperimentRequestId).catch((e) => {
      console.error(e);
    });

    this.handleGettingRuns(this.props.searchRunsApi, this.searchRunsRequestId);
  }

  maybeReloadData(prevProps) {
    if (this.props.experimentId !== prevProps.experimentId) {
      this.loadData();
    }
  }

  updateNumRunsFromLatestSearch = (response = {}) => {
    const { value } = response;
    if (value && value.runs) {
      this.setState({ numRunsFromLatestSearch: response.value.runs.length });
    }
    return response;
  };

  updateNextPageToken = (response = {}) => {
    const { value } = response;
    let nextPageToken = null;
    if (value && value.next_page_token) {
      nextPageToken = value.next_page_token;
    }
    this.setState({ nextPageToken, loadingMore: false });
    return response;
  };

  fetchModelVersionsForRuns = (response = {}) => {
    const { value } = response;
    if (value) {
      const { runs } = value;
      if (runs && runs.length > 0) {
        _.chunk(runs, MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER).forEach((runsChunk) => {
          this.props.searchModelVersionsApi(
            { run_id: runsChunk.map((run) => run.info.run_id) },
            this.searchModelVersionsRequestId,
          );
        });
      }
    }

    return response;
  };

  static StartTimeColumnOffset = {
    ALL: null,
    LAST_HOUR: 1 * 60 * 60 * 1000,
    LAST_24_HOURS: 24 * 60 * 60 * 1000,
    LAST_7_DAYS: 7 * 24 * 60 * 60 * 1000,
    LAST_30_DAYS: 30 * 24 * 60 * 60 * 1000,
    LAST_YEAR: 12 * 30 * 24 * 60 * 60 * 1000,
  };

  getStartTimeExpr() {
    const startTimeColumnOffset = ExperimentPage.StartTimeColumnOffset;
    const { startTime } = this.state.persistedState;
    const offset = startTimeColumnOffset[startTime];
    if (!startTime || !offset || startTime === 'ALL') {
      return null;
    }
    const startTimeOffset = new Date() - offset;

    return `attributes.start_time >= ${startTimeOffset}`;
  }

  handleGettingRuns = (getRunsAction, requestId) => {
    const { persistedState, lifecycleFilter, nextPageToken } = this.state;
    const { searchInput } = persistedState;
    const viewType = lifecycleFilterToRunViewType(lifecycleFilter);
    const orderBy = this.getOrderByExpr();
    const startTime = this.getStartTimeExpr();
    let filter = searchInput;
    if (startTime) {
      if (filter.length > 0) {
        filter = `${filter} && ${startTime}`;
      } else {
        filter = startTime;
      }
    }
    const shouldFetchParents = this.shouldNestChildrenAndFetchParents();
    return getRunsAction({
      filter,
      runViewType: viewType,
      experimentIds: [this.props.experimentId],
      orderBy,
      pageToken: nextPageToken,
      shouldFetchParents,
      id: requestId,
    })
      .then(this.updateNextPageToken)
      .then(this.updateNumRunsFromLatestSearch)
      .then(this.fetchModelVersionsForRuns)
      .catch((e) => {
        Utils.logErrorAndNotifyUser(e);
        this.setState({ ...PAGINATION_DEFAULT_STATE });
      });
  };

  handleLoadMoreRuns = () => {
    this.setState({ loadingMore: true });
    this.handleGettingRuns(this.props.loadMoreRunsApi, this.loadMoreRunsRequestId);
  };

  /*
    If this function returns true, the ExperimentView should nest children underneath their parents
    and fetch all root level parents of visible runs. If this function returns false, the views will
    not nest children or fetch any additional parents. Will always return true if the orderByKey is
    'attributes.start_time'
  */
  shouldNestChildrenAndFetchParents() {
    const { orderByKey, searchInput } = this.state.persistedState;
    return (!orderByKey && !searchInput) || orderByKey === ATTRIBUTE_COLUMN_SORT_KEY.DATE;
  }

  handleColumnSelectionCheck = (categorizedUncheckedKeys) => {
    this.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          ...this.state.persistedState,
          categorizedUncheckedKeys,
        }).toJSON(),
      },
      () => {
        this.updateUrlWithSearchFilter({ categorizedUncheckedKeys });
      },
    );
  };

  onSearch = (
    paramKeyFilterString,
    metricKeyFilterString,
    searchInput,
    lifecycleFilterInput,
    orderByKey,
    orderByAsc,
    modelVersionFilterInput,
    startTime,
    categorizedUncheckedKeys,
  ) => {
    this.updateUrlWithSearchFilter({
      paramKeyFilterString,
      metricKeyFilterString,
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      categorizedUncheckedKeys,
    });

    this.setState(
      {
        lastRunsRefreshTime: Date.now(),
        numberOfNewRuns: 0,
        persistedState: new ExperimentPagePersistedState({
          paramKeyFilterString,
          metricKeyFilterString,
          searchInput,
          orderByKey,
          orderByAsc,
          startTime,
          categorizedUncheckedKeys,
        }).toJSON(),
        lifecycleFilter: lifecycleFilterInput,
        modelVersionFilter: modelVersionFilterInput,
        nextPageToken: null,
      },
      () => {
        this.handleGettingRuns(this.props.searchRunsApi, this.searchRunsRequestId);
        if (!this.detectNewRunsTimer) {
          this.detectNewRunsTimer = setInterval(
            () => this.detectNewRuns(),
            DETECT_NEW_RUNS_INTERVAL,
          );
        }
      },
    );
  };

  async detectNewRuns() {
    if (Utils.isBrowserTabVisible()) {
      const lastRunsRefreshTime = this.state.lastRunsRefreshTime || 0;
      const latestRuns = await this.props.searchForNewRuns({
        experimentIds: [this.props.experimentId],
        maxResults: MAX_DETECT_NEW_RUNS_RESULTS,
      });
      let numberOfNewRuns = 0;
      if (latestRuns && latestRuns.runs) {
        numberOfNewRuns = latestRuns.runs.filter((run) => isNewRun(lastRunsRefreshTime, run))
          .length;

        if (numberOfNewRuns >= MAX_DETECT_NEW_RUNS_RESULTS) {
          clearInterval(this.detectNewRunsTimer);
          this.detectNewRunsTimer = null;
        }
      }

      this.setState((previousState) => {
        if (previousState.numberOfNewRuns !== numberOfNewRuns) {
          return { numberOfNewRuns };
        }

        // Don't re-render the component if the state is exactly the same
        return null;
      });
    }
  }

  getOrderByExpr() {
    const { orderByKey, orderByAsc } = this.state.persistedState;
    let orderBy = [];
    if (orderByKey) {
      if (orderByAsc) {
        orderBy = [orderByKey + ' ASC'];
      } else {
        orderBy = [orderByKey + ' DESC'];
      }
    }
    return orderBy;
  }

  updateUrlWithSearchFilter({
    paramKeyFilterString,
    metricKeyFilterString,
    searchInput,
    orderByKey,
    orderByAsc,
    startTime,
    categorizedUncheckedKeys,
  }) {
    const state = {};
    if (paramKeyFilterString || this.state.persistedState.paramKeyFilterString) {
      state['params'] = paramKeyFilterString || this.state.persistedState.paramKeyFilterString;
    }
    if (metricKeyFilterString || this.state.persistedState.metricKeyFilterString) {
      state['metrics'] = metricKeyFilterString || this.state.persistedState.metricKeyFilterString;
    }
    if (searchInput || this.state.persistedState.searchInput) {
      state['search'] = searchInput || this.state.persistedState.searchInput;
    }
    if (startTime || this.state.persistedState.startTime) {
      state['startTime'] = startTime || this.state.persistedState.startTime;
    }
    if (orderByKey || this.state.persistedState.orderByKey) {
      state['orderByKey'] = orderByKey || this.state.persistedState.orderByKey;
    }
    // orderByAsc defaults to true, so only encode it if it is false.
    if (orderByAsc === false || this.state.persistedState.orderByAsc === false) {
      state['orderByAsc'] = orderByAsc || this.state.persistedState.orderByAsc;
    }
    if (categorizedUncheckedKeys || this.state.persistedState.categorizedUncheckedKeys) {
      state['categorizedUncheckedKeys'] =
        categorizedUncheckedKeys || this.state.persistedState.categorizedUncheckedKeys;
    }
    const newUrl = `/experiments/${this.props.experimentId}/s?${Utils.getSearchUrlFromState(
      state,
    )}`;
    if (newUrl !== this.props.history.location.pathname + this.props.history.location.search) {
      this.props.history.push(newUrl);
    }
  }

  renderExperimentView = (isLoading, shouldRenderError, requests) => {
    let searchRunsError;
    const getExperimentRequest = Utils.getRequestWithId(requests, this.getExperimentRequestId);

    if (shouldRenderError) {
      const searchRunsRequest = Utils.getRequestWithId(requests, this.searchRunsRequestId);
      if (
        getExperimentRequest.error &&
        getExperimentRequest.error.getErrorCode() === ErrorCodes.PERMISSION_DENIED
      ) {
        return <PermissionDeniedView errorMessage={getExperimentRequest.error.getMessageField()} />;
      } else if (searchRunsRequest.error) {
        searchRunsError = searchRunsRequest.error.getMessageField();
      } else {
        return undefined;
      }
    }
    if (!getExperimentRequest || getExperimentRequest.active) {
      return <Spinner />;
    }

    const {
      paramKeyFilterString,
      metricKeyFilterString,
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      categorizedUncheckedKeys,
    } = this.state.persistedState;

    const experimentViewProps = {
      paramKeyFilter: new KeyFilter(paramKeyFilterString),
      metricKeyFilter: new KeyFilter(metricKeyFilterString),
      experimentId: this.props.experimentId,
      experiment: this.props.experiment,
      searchRunsRequestId: this.searchRunsRequestId,
      modelVersionFilter: this.state.modelVersionFilter,
      lifecycleFilter: this.state.lifecycleFilter,
      onSearch: this.onSearch,
      searchRunsError: searchRunsError,
      searchInput: searchInput,
      isLoading: isLoading && !searchRunsError,
      orderByKey: orderByKey,
      orderByAsc: orderByAsc,
      startTime: startTime,
      categorizedUncheckedKeys: categorizedUncheckedKeys,
      nextPageToken: this.state.nextPageToken,
      numRunsFromLatestSearch: this.state.numRunsFromLatestSearch,
      handleLoadMoreRuns: this.handleLoadMoreRuns,
      handleColumnSelectionCheck: this.handleColumnSelectionCheck.bind(this),
      loadingMore: this.state.loadingMore,
      nestChildren: this.shouldNestChildrenAndFetchParents(orderByKey, searchInput),
      numberOfNewRuns: this.state.numberOfNewRuns,
    };

    return <ExperimentView {...experimentViewProps} />;
  };

  render() {
    return (
      <div className='ExperimentPage runs-table-flex-container' style={{ height: '100%' }}>
        <RequestStateWrapper shouldOptimisticallyRender requestIds={this.getRequestIds()}>
          {this.renderExperimentView}
        </RequestStateWrapper>
      </div>
    );
  }

  getRequestIds() {
    return [this.getExperimentRequestId, this.searchRunsRequestId];
  }
}

const mapStateToProps = (state, ownProps) => {
  const experiment = getExperiment(ownProps.experimentId, state);
  return { experiment };
};

const mapDispatchToProps = {
  getExperimentApi,
  searchRunsApi,
  loadMoreRunsApi,
  searchModelVersionsApi,
};

export const lifecycleFilterToRunViewType = (lifecycleFilter) => {
  if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
    return ViewType.ACTIVE_ONLY;
  } else {
    return ViewType.DELETED_ONLY;
  }
};

export default withRouter(connect(mapStateToProps, mapDispatchToProps)(injectIntl(ExperimentPage)));
