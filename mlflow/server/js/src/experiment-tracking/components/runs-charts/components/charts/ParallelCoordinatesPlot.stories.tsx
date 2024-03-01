import { useCallback, useState } from 'react';
import { ChartStoryWrapper, useControls } from '../RunsCharts.stories-common';
import LazyParallelCoordinatesPlot from './LazyParallelCoordinatesPlot';
import './ParallelCoordinatesPlot.css';

export default {
  title: 'Parallel Coordinates Plot',
  component: LazyParallelCoordinatesPlot,
  argTypes: {},
  parameters: {
    layout: 'fullscreen',
  },
};

const data = [
  {
    metric_0: 2,
    metric_1: 10,
    metric_2: 16,
    metric_3: 19,
    metric_4: 18,
    metric_5: 0,
    metric_6: 9,
    metric_7: 15,
    metric_8: 16,
    metric_9: 2,
    param_0: 'cherry',
    param_1: 'banana',
    primary_metric: 0.9160702935526702,
    uuid: '454bfb22-0537-4ebf-826f-56bc2bd7c59d',
  },
  {
    metric_0: 0,
    metric_1: 0,
    metric_2: 7,
    metric_3: 5,
    metric_4: 16,
    metric_5: 7,
    metric_6: 6,
    metric_7: 5,
    metric_8: 8,
    metric_9: 9,
    param_0: 'apple',
    param_1: 'cherry',
    primary_metric: 0.015731080795666874,
    uuid: '2b8515bd-5e42-4d0d-9ac3-ff495c5e1fef',
  },
  {
    metric_0: 0,
    metric_1: 6,
    metric_2: 17,
    metric_3: 11,
    metric_4: 20,
    metric_5: 6,
    metric_6: 1,
    metric_7: 0,
    metric_8: 16,
    metric_9: 12,
    param_0: 'banana',
    param_1: 'banana',
    primary_metric: 0.251069539401114,
    uuid: '0b9347d4-24ed-4be0-8368-855ad9eb82e6',
  },
  {
    metric_0: 3,
    metric_1: 18,
    metric_2: 20,
    metric_3: 18,
    metric_4: 7,
    metric_5: 7,
    metric_6: 5,
    metric_7: 13,
    metric_8: 7,
    metric_9: 10,
    param_0: 'cherry',
    param_1: 'cherry',
    primary_metric: 0.32246109284238333,
    uuid: 'a9074b60-5e1e-4de8-87ec-43cde5928f1c',
  },
  {
    metric_0: 13,
    metric_1: 17,
    metric_2: 6,
    metric_3: 9,
    metric_4: 1,
    metric_5: 3,
    metric_6: 16,
    metric_7: 6,
    metric_8: 19,
    metric_9: 0,
    param_0: 'cherry',
    param_1: 'banana',
    primary_metric: 0.9459501114598016,
    uuid: 'bd68d58d-8f40-4b9a-b81b-cb3524c3ecc1',
  },
  {
    metric_0: 2,
    metric_1: 6,
    metric_2: 17,
    metric_3: 9,
    metric_4: 2,
    metric_5: 3,
    metric_6: 5,
    metric_7: 15,
    metric_8: 16,
    metric_9: 7,
    param_0: 'apple',
    param_1: 'apple',
    primary_metric: 0.9669168287904518,
    uuid: 'a3f2a208-3055-4505-8c97-ba9de34b6568',
  },
  {
    metric_0: 12,
    metric_1: 8,
    metric_2: 17,
    metric_3: 0,
    metric_4: 14,
    metric_5: 12,
    metric_6: 14,
    metric_7: 8,
    metric_8: 15,
    metric_9: 9,
    param_0: 'banana',
    param_1: 'banana',
    primary_metric: 0.36678200892298674,
    uuid: 'eae6130e-ca94-44ff-9177-adead56f8354',
  },
  {
    metric_0: 10,
    metric_1: 4,
    metric_2: 7,
    metric_3: 15,
    metric_4: 5,
    metric_5: 8,
    metric_6: 15,
    metric_7: 5,
    metric_8: 12,
    metric_9: 8,
    param_0: 'apple',
    param_1: 'banana',
    primary_metric: 0.4045235357908987,
    uuid: '64299111-da28-4a24-ad33-07158696a393',
  },
  {
    metric_0: 4,
    metric_1: 12,
    metric_2: 3,
    metric_3: 7,
    metric_4: 19,
    metric_5: 17,
    metric_6: 10,
    metric_7: 11,
    metric_8: 15,
    metric_9: 11,
    param_0: 'cherry',
    param_1: 'banana',
    primary_metric: 0.531512209049746,
    uuid: '460cc8a4-52e4-40dc-ab80-ef30d2254aa2',
  },
  {
    metric_0: 4,
    metric_1: 14,
    metric_2: 0,
    metric_3: 0,
    metric_4: 5,
    metric_5: 20,
    metric_6: 9,
    metric_7: 8,
    metric_8: 19,
    metric_9: 15,
    param_0: 'banana',
    param_1: 'cherry',
    primary_metric: 0.7763796146588676,
    uuid: 'c4a61f7b-f793-4d76-8b99-36e74df2fc04',
  },
  {
    metric_0: 16,
    metric_1: 13,
    metric_2: 20,
    metric_3: 15,
    metric_4: 18,
    metric_5: 9,
    metric_6: 8,
    metric_7: 18,
    metric_8: 6,
    metric_9: 6,
    param_0: 'cherry',
    param_1: 'banana',
    primary_metric: 0.483225391587373,
    uuid: 'e58bc3e5-a419-45e6-bffd-38b846564d17',
  },
  {
    metric_0: 10,
    metric_1: 2,
    metric_2: 6,
    metric_3: 12,
    metric_4: 14,
    metric_5: 10,
    metric_6: 11,
    metric_7: 18,
    metric_8: 0,
    metric_9: 9,
    param_0: 'apple',
    param_1: 'apple',
    primary_metric: 0.18085852097860733,
    uuid: '1e591818-1a3e-406c-affc-3fd37bf2f48f',
  },
  {
    metric_0: 5,
    metric_1: 15,
    metric_2: 19,
    metric_3: 4,
    metric_4: 9,
    metric_5: 14,
    metric_6: 15,
    metric_7: 1,
    metric_8: 13,
    metric_9: 3,
    param_0: 'cherry',
    param_1: 'banana',
    primary_metric: 0.29352171511799907,
    uuid: '6e889e10-25d3-4795-a4ab-6d856ed19fc4',
  },
  {
    metric_0: 7,
    metric_1: 0,
    metric_2: 13,
    metric_3: 12,
    metric_4: 4,
    metric_5: 18,
    metric_6: 20,
    metric_7: 4,
    metric_8: 1,
    metric_9: 1,
    param_0: 'banana',
    param_1: 'apple',
    primary_metric: 0.9668485150687122,
    uuid: '2c273da0-70a9-4f2e-a287-4c403fd53d57',
  },
  {
    metric_0: 5,
    metric_1: 9,
    metric_2: 1,
    metric_3: 17,
    metric_4: 11,
    metric_5: 11,
    metric_6: 16,
    metric_7: 9,
    metric_8: 18,
    metric_9: 20,
    param_0: 'cherry',
    param_1: 'cherry',
    primary_metric: 0.27369675268567917,
    uuid: 'd1381c10-e204-4c95-84bd-71b9c995e7d9',
  },
  {
    metric_0: 15,
    metric_1: 13,
    metric_2: 20,
    metric_3: 7,
    metric_4: 2,
    metric_5: 11,
    metric_6: 3,
    metric_7: 18,
    metric_8: 3,
    metric_9: 0,
    param_0: 'apple',
    param_1: 'apple',
    primary_metric: 0.9345787375230612,
    uuid: 'b7584816-15c2-44d7-8205-dfa0b0298199',
  },
  {
    metric_0: 11,
    metric_1: 14,
    metric_2: 14,
    metric_3: 1,
    metric_4: 19,
    metric_5: 6,
    metric_6: 3,
    metric_7: 2,
    metric_8: 4,
    metric_9: 10,
    param_0: 'banana',
    param_1: 'banana',
    primary_metric: 0.785438155859252,
    uuid: '85c24ea0-203f-49da-b015-bdf204223df3',
  },
  {
    metric_0: 12,
    metric_1: 0,
    metric_2: 13,
    metric_3: 9,
    metric_4: 18,
    metric_5: 14,
    metric_6: 16,
    metric_7: 3,
    metric_8: 14,
    metric_9: 10,
    param_0: 'banana',
    param_1: 'apple',
    primary_metric: 0.7789893653419077,
    uuid: '0335792d-7792-4d64-8587-18e144d3ad71',
  },
  {
    metric_0: 20,
    metric_1: 8,
    metric_2: 9,
    metric_3: 7,
    metric_4: 11,
    metric_5: 11,
    metric_6: 12,
    metric_7: 13,
    metric_8: 13,
    metric_9: 4,
    param_0: 'banana',
    param_1: 'cherry',
    primary_metric: 0.566828475655832,
    uuid: '1750f839-6ef4-41b6-9692-35509d46de50',
  },
  {
    metric_0: 18,
    metric_1: 15,
    metric_2: 12,
    metric_3: 1,
    metric_4: 4,
    metric_5: 14,
    metric_6: 17,
    metric_7: 3,
    metric_8: 10,
    metric_9: 18,
    param_0: 'cherry',
    param_1: 'cherry',
    primary_metric: 0.04427974889292041,
    uuid: '60f8294f-ad58-4cc0-bbcb-a2066dff3ad2',
  },
];

const ParallelCoordinatesPlotStoryWrapper = (props: any) => {
  const { axisProps, controls } = useControls(false);
  const [hoveredRun, setHoveredRun] = useState('');

  const clear = useCallback(() => setHoveredRun(''), []);

  return (
    <ChartStoryWrapper
      title={props.title}
      controls={
        <>
          {controls}
          Hovered run ID: {hoveredRun}
        </>
      }
    >
      <LazyParallelCoordinatesPlot {...axisProps} onHover={setHoveredRun} onUnhover={clear} {...props} />
    </ChartStoryWrapper>
  );
};

export const ParallelCoords = () => <ParallelCoordinatesPlotStoryWrapper data={data} />;

ParallelCoords.storyName = 'Parallel Coordinates Plot';