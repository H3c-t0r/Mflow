import { useCallback, useEffect, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { GenericSkeleton, Input, Modal } from '@databricks/design-system';
import { useDispatch } from 'react-redux';
import { ThunkDispatch } from '../../../../../redux-types';
import { setExperimentTagApi } from '../../../../actions';
import Routes from '../../../../routes';
import { CopyButton } from '../../../../../shared/building_blocks/CopyButton';
import { ExperimentPageSearchFacetsStateV2 } from '../../../experiment-page/models/ExperimentPageSearchFacetsStateV2';
import { ExperimentPageUIStateV2 } from '../../../experiment-page/models/ExperimentPageUIStateV2';
import { getStringSHA256 } from '../../../../../common/utils/StringUtils';
import Utils from '../../../../../common/utils/Utils';
import { EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX } from '../../../../constants';

type GetShareLinkModalProps = {
  onCancel: () => void;
  visible: boolean;
  experimentIds: string[];
  searchFacetsState: ExperimentPageSearchFacetsStateV2;
  uiState: ExperimentPageUIStateV2;
};

type ShareableViewState = ExperimentPageSearchFacetsStateV2 & ExperimentPageUIStateV2;

/**
 * Modal that displays shareable link for the experiment page.
 * The shareable state is created by serializing the search facets and UI state and storing
 * it as a tag on the experiment.
 */
export const ExperimentGetShareLinkModal = ({
  onCancel,
  visible,
  experimentIds,
  searchFacetsState,
  uiState,
}: GetShareLinkModalProps) => {
  const [sharedStateUrl, setSharedStateUrl] = useState<string>('');
  const [linkInProgress, setLinkInProgress] = useState(true);
  const [generatedState, setGeneratedState] = useState<ShareableViewState | null>(null);

  const dispatch = useDispatch<ThunkDispatch>();

  const stateToSerialize = useMemo(() => ({ ...searchFacetsState, ...uiState }), [searchFacetsState, uiState]);

  const createSerializedState = useCallback(
    async (state: ShareableViewState) => {
      if (experimentIds.length > 1) {
        setLinkInProgress(false);
        setGeneratedState(state);
        setSharedStateUrl(window.location.href);
        return;
      }
      setLinkInProgress(true);
      const [experimentId] = experimentIds;
      const data = JSON.stringify(state);
      const hash = await getStringSHA256(data);

      const tagName = `${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}${hash}`;

      dispatch(setExperimentTagApi(experimentId, tagName, data))
        .then(() => {
          setLinkInProgress(false);
          setGeneratedState(state);
          const pageRoute = Routes.getExperimentPageRoute(experimentId, false, hash);
          const shareURL = `${window.location.origin}${window.location.pathname}#${pageRoute}`;
          setSharedStateUrl(shareURL);
        })
        .catch((e) => {
          Utils.logErrorAndNotifyUser('Failed to create shareable link for experiment');
          throw e;
        });
    },
    [dispatch, experimentIds],
  );

  useEffect(() => {
    if (!visible || generatedState === stateToSerialize) {
      return;
    }
    createSerializedState(stateToSerialize);
  }, [visible, createSerializedState, generatedState, stateToSerialize]);

  return (
    <Modal
      title={
        <FormattedMessage
          defaultMessage="Get shareable link"
          description={'Title text for the experiment "Get link" modal'}
        />
      }
      visible={visible}
      onCancel={onCancel}
    >
      <div css={{ display: 'flex', gap: 8 }}>
        {linkInProgress ? (
          <GenericSkeleton css={{ flex: 1 }} />
        ) : (
          <Input placeholder="Click button on the right to create shareable state" value={sharedStateUrl} readOnly />
        )}
        <CopyButton loading={linkInProgress} copyText={sharedStateUrl} />
      </div>
    </Modal>
  );
};
