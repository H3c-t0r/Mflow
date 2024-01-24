import { useMemo } from 'react';
import { RunsMetricsBarPlot } from '../runs-charts/components/RunsMetricsBarPlot';
import { useDesignSystemTheme } from '@databricks/design-system';
import { useRunsChartsTooltip } from '../runs-charts/hooks/useRunsChartsTooltip';
import { MetricEntity, RunInfoEntity } from '../../types';

export interface RunViewSingleMetricChartProps {
  metricKey: string;
  runInfo: RunInfoEntity;
  metricEntry?: MetricEntity;
}

/**
 * Chart variant displaying single (non-history) value using bar plot
 */
export const RunViewMetricSingleValueChart = ({ runInfo, metricKey, metricEntry }: RunViewSingleMetricChartProps) => {
  const { theme } = useDesignSystemTheme();
  const { setTooltip, resetTooltip } = useRunsChartsTooltip({ metricKey });

  // Prepare a single trace for the line chart
  const chartData = useMemo(
    () =>
      metricEntry
        ? [
            {
              uuid: runInfo.run_uuid,
              displayName: runInfo.run_name,
              runInfo,
              metrics: { [metricKey]: metricEntry },
              color: theme.colors.primary,
            },
          ]
        : [],
    [runInfo, metricEntry, metricKey, theme],
  );

  return (
    <RunsMetricsBarPlot
      metricKey={metricKey}
      runsData={chartData}
      height={260}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      useDefaultHoverBox={false}
      displayRunNames={false}
      displayMetricKey={false}
    />
  );
};
