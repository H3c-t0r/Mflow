/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import './ModelListView.css';
import Utils from '../../common/utils/Utils';
import {
  REGISTERED_MODELS_PER_PAGE_COMPACT,
  REGISTERED_MODELS_SEARCH_NAME_FIELD,
  REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD,
} from '../constants';
import {
  ModelRegistryDocUrl,
  ModelRegistryOnboardingString,
  onboarding,
} from '../../common/constants';
import { CreateModelButton } from './CreateModelButton';
import LocalStorageUtils from '../../common/utils/LocalStorageUtils';
import { PageHeader } from '../../shared/building_blocks/PageHeader';
import { FormattedMessage, injectIntl } from 'react-intl';
import {
  Alert,
  CursorPagination,
  InfoIcon,
  LegacyPopover,
  Spacer as DuBoisSpacer,
} from '@databricks/design-system';
import { ModelListFilters } from './model-list/ModelListFilters';
import { ModelListTable } from './model-list/ModelListTable';
import { PageContainer } from '../../common/components/PageContainer';

const NAME_COLUMN_INDEX = 'name';
const LAST_MODIFIED_COLUMN_INDEX = 'last_updated_timestamp';

type OwnModelListViewImplProps = {
  models: any[];
  endpoints?: any;
  showEditPermissionModal: (...args: any[]) => any;
  permissionLevel: string;
  selectedOwnerFilter: string;
  selectedStatusFilter: string;
  onOwnerFilterChange: (...args: any[]) => any;
  onStatusFilterChange: (...args: any[]) => any;
  searchInput: string;
  orderByKey: string;
  orderByAsc: boolean;
  currentPage: number;
  nextPageToken?: string;
  loading?: boolean;
  onSearch: (...args: any[]) => any;
  onClickNext: (...args: any[]) => any;
  onClickPrev: (...args: any[]) => any;
  onClickSortableColumn: (...args: any[]) => any;
  onSetMaxResult: (...args: any[]) => any;
  getMaxResultValue: (...args: any[]) => any;
  intl?: any;
};

type ModelListViewImplState = any;

type ModelListViewImplProps = OwnModelListViewImplProps & typeof ModelListViewImpl.defaultProps;

export class ModelListViewImpl extends React.Component<
  ModelListViewImplProps,
  ModelListViewImplState
> {
  constructor(props: ModelListViewImplProps) {
    super(props);

    this.state = {
      loading: false,
      lastNavigationActionWasClickPrev: false,
      maxResultsSelection: REGISTERED_MODELS_PER_PAGE_COMPACT,
      showOnboardingHelper: this.showOnboardingHelper(),
    };
  }

  static defaultProps = {
    models: [],
    searchInput: '',
  };

  showOnboardingHelper() {
    const onboardingInformationStore = ModelListViewImpl.getLocalStore(onboarding);
    return onboardingInformationStore.getItem('showRegistryHelper') === null;
  }

  disableOnboardingHelper() {
    const onboardingInformationStore = ModelListViewImpl.getLocalStore(onboarding);
    onboardingInformationStore.setItem('showRegistryHelper', 'false');
  }

  /**
   * Returns a LocalStorageStore instance that can be used to persist data associated with the
   * ModelRegistry component.
   */
  static getLocalStore(key: any) {
    return LocalStorageUtils.getStoreForComponent('ModelListView', key);
  }

  componentDidMount() {
    const pageTitle = 'MLflow Models';
    Utils.updatePageTitle(pageTitle);
  }

  setLoadingFalse = () => {
    this.setState({ loading: false });
  };

  handleSearch = (event: any, searchInput: any) => {
    event?.preventDefault();
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onSearch(this.setLoadingFalse, this.setLoadingFalse, searchInput);
  };

  static getSortFieldName = (column: any) => {
    switch (column) {
      case NAME_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_NAME_FIELD;
      case LAST_MODIFIED_COLUMN_INDEX:
        return REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD;
      default:
        return null;
    }
  };

  unifiedTableSortChange = ({ orderByKey, orderByAsc }: any) => {
    // Different column keys are used for sorting and data accessing,
    // mapping to proper keys happens below
    const fieldMappedToSortKey =
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      {
        timestamp: 'last_updated_timestamp',
      }[orderByKey] || orderByKey;

    this.handleTableChange(undefined, undefined, {
      field: fieldMappedToSortKey,
      order: orderByAsc ? 'undefined' : 'descend',
    });
  };

  handleTableChange = (pagination: any, filters: any, sorter: any) => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onClickSortableColumn(
      ModelListViewImpl.getSortFieldName(sorter.field),
      sorter.order,
      this.setLoadingFalse,
      this.setLoadingFalse,
    );
  };

  static getLearnMoreLinkUrl = () => ModelRegistryDocUrl;

  static getLearnMoreDisplayString = () => ModelRegistryOnboardingString;

  handleClickNext = () => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: false });
    this.props.onClickNext(this.setLoadingFalse, this.setLoadingFalse);
  };

  handleClickPrev = () => {
    this.setState({ loading: true, lastNavigationActionWasClickPrev: true });
    this.props.onClickPrev(this.setLoadingFalse, this.setLoadingFalse);
  };

  handleSetMaxResult = ({ item, key, keyPath, domEvent }: any) => {
    this.setState({ loading: true });
    this.props.onSetMaxResult(key, this.setLoadingFalse, this.setLoadingFalse);
  };

  render() {
    // prettier-ignore
    const {
      models,
      currentPage,
      nextPageToken,
      searchInput,
    } = this.props;
    const { loading, showOnboardingHelper } = this.state;

    // Determine if we use any filters at the moment
    const isFiltered =
      // prettier-ignore
      Boolean(searchInput);

    const title = (
      <FormattedMessage
        defaultMessage='Registered Models'
        description='Header for displaying models in the model registry'
      />
    );
    return (
      <PageContainer data-test-id='ModelListView-container' usesFullHeight>
        <div>
          <PageHeader
            title={
              <>
                {title}{' '}
                <LegacyPopover
                  content={
                    // this showOnboardingHelper may not be necessary.
                    showOnboardingHelper && (
                      <>
                        {ModelListViewImpl.getLearnMoreDisplayString()}{' '}
                        <FormattedMessage
                          defaultMessage='<link>Learn more</link>'
                          description='Learn more link on the model list page with cloud-specific link'
                          values={{
                            link: (chunks: any) => (
                              <a
                                href={ModelListViewImpl.getLearnMoreLinkUrl()}
                                target='_blank'
                                rel='noopener noreferrer'
                                className='LinkColor'
                              >
                                {chunks}
                              </a>
                            ),
                          }}
                        />
                      </>
                    )
                  }
                >
                  <InfoIcon css={{ cursor: 'pointer' }} />
                </LegacyPopover>
              </>
            }
          >
            <CreateModelButton />
          </PageHeader>
          <ModelListFilters
            searchFilter={this.props.searchInput}
            onSearchFilterChange={(value) => this.handleSearch(null, value)}
            isFiltered={isFiltered}
          />
        </div>
        <ModelListTable
          modelsData={models}
          onSortChange={this.unifiedTableSortChange}
          orderByKey={this.props.orderByKey}
          orderByAsc={this.props.orderByAsc}
          isLoading={loading}
          pagination={
            <div data-testid='model-list-view-pagination'>
              <CursorPagination
                hasNextPage={Boolean(nextPageToken)}
                hasPreviousPage={currentPage > 1}
                onNextPage={this.handleClickNext}
                onPreviousPage={this.handleClickPrev}
                pageSizeSelect={{
                  onChange: (num) => this.handleSetMaxResult({ key: num }),
                  default: this.props.getMaxResultValue(),
                  options: [10, 25, 50, 100],
                }}
              />
            </div>
          }
          isFiltered={isFiltered}
        />
      </PageContainer>
    );
  }
}

// @ts-expect-error TS(2769): No overload matches this call.
export const ModelListView = injectIntl(ModelListViewImpl);

const styles = {
  nameSearchBox: {
    width: '446px',
  },
  searchFlexBar: {
    marginBottom: '24px',
  },
  questionMark: {
    marginLeft: 4,
    cursor: 'pointer',
    color: '#888',
  },
};
