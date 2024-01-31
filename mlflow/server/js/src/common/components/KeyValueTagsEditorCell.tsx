import { Button, PencilIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { KeyValueEntity } from '../../experiment-tracking/types';
import { KeyValueTag } from './KeyValueTag';

interface KeyValueTagsEditorCellProps {
  tags?: KeyValueEntity[];
  onAddEdit: () => void;
}

/**
 * A cell renderer used in tables, displaying a list of key-value tags with button for editing those
 */
export const KeyValueTagsEditorCell = ({ tags = [], onAddEdit }: KeyValueTagsEditorCellProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexWrap: 'wrap',
        '> *': {
          marginRight: '0 !important',
        },
        gap: theme.spacing.xs,
      }}
    >
      {tags.length < 1 ? (
        <Button size="small" type="link" onClick={onAddEdit}>
          <FormattedMessage defaultMessage="Add" description="Key-value tag table cell > 'add' button label" />
        </Button>
      ) : (
        <>
          {tags.map((tag) => (
            <KeyValueTag tag={tag} key={`${tag.key}-${tag.value}`} />
          ))}
          <Button size="small" icon={<PencilIcon />} onClick={onAddEdit} />
        </>
      )}
    </div>
  );
};
