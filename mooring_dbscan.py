import warnings

import shapely

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from shapely.ops import unary_union
import utm
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from sklearn.cluster import DBSCAN
import config as cfg
import pymonetdb
import skmob
from skmob.preprocessing.detection import stay_locations
from scipy.stats import circmean


class DataObject:
    def __init__(self):
        if not cfg.LOCAL_FILE:
            self.connection = pymonetdb.connect(username=cfg.MONETDB_USERNAME, password=cfg.MONETDB_PASSWORD,
                                                hostname=cfg.DB_URL, database=cfg.DB_NAME)
        self.data = None
        self.train = None
        self.clustering_results = None
        self.load_data()

    def load_data(self):
        """
        Method that loads the AIS data from either local csv file or a MonetDB database
        :param local_data: whether the data is stored locally
        """
        if not cfg.LOCAL_FILE:
            self.fetch_data_db()
        else:
            self.fetch_data_local()

    def fetch_data_local(self):
        """
        Reads the data from csv and changes the column names to correct format
        """
        df = pd.read_csv(cfg.LOCAL_FILE)
        self.data = df.rename(columns=cfg.COLUMN_NAMES)

    def fetch_data_db(self):
        """
        Fetch data from database based on year and area

        :param year: timeframe of the data
        :param port: name of port
        :return: df: AIS data in pandas dataframe format
        """
        cursor = self.connection.cursor()
        cursor.arraysize = 10000
        # Create a mask for the bounding box?
        # Read from WPI to get the coordinates for the port?
        _ = cursor.execute(
            f'SELECT {cfg.COLUMN_NAMES["mmsi"]}, {cfg.COLUMN_NAMES["status"]}, {cfg.COLUMN_NAMES["speed"]}, {cfg.COLUMN_NAMES["heading"]},'
            f'{cfg.COLUMN_NAMES["lon"]}, {cfg.COLUMN_NAMES["lat"]}, {cfg.COLUMN_NAMES["time"]} FROM brest_dynamic')  # WHERE shiptype BETWEEN 70 AND 89')
        self.connection.commit()
        self.data = pd.DataFrame(cursor.fetchall(),
                                 columns=['mmsi', 'status', 'speed', 'heading', 'lon', 'lat', 'time'])

    def preprocess(self):
        """
        Changes the data to trajectory dataframe and removes all ships with less than 100 messages in the data
        """
        self.data['datetime'] = pd.to_datetime(self.data.t, unit='s')
        self.data = skmob.TrajDataFrame(self.data, longitude='lon', user_id='sourcemmsi', )
        self.data = self.data[self.data.trueheading <= 360]
        ships = self.data.groupby('uid').size() > 100
        self.data = self.data[self.data.uid.isin(ships[ships].index)]

    def detect_groups(self):
        """
        Extracts stop locations from the data. A stop is defined as a ship staying in one place for atleast 2 hours
        """
        stops = stay_locations(self.data, minutes_for_a_stop=120, spatial_radius_km=0.001, no_data_for_minutes=60,
                               min_speed_kmh=0.5)
        self.train = stops.to_geodataframe()

    def run(self):
        """
        Runs all the steps of this class in sequential order
        """
        self.preprocess()
        self.detect_groups()

    def __del__(self):
        if not cfg.LOCAL_FILE:
            self.connection.close()


class Model:
    def __init__(self):
        self.utm_zone = None
        self.train_data = None
        self.combined_quays = None
        self.algorithm = DBSCAN
        self.clustering_results = None

    def set_data(self, data: gpd.GeoDataFrame) -> object:
        """
        Set the training data for the model
        :param data: coordinate data in WGS84 format
        """
        self.train_data = self.change_coordinates_to_utm(data)

    def run(self, data: gpd.GeoDataFrame) -> object:
        """
        Run the whole modeling pipeline
        :param data: training data in WGS84 format
        """
        self.set_data(data)
        self.train_model()
        self.create_polygons()
        self.filter_polygons()
        self.find_quays()
        self.clustering_results.to_crs(4326, inplace=True)

    def _get_utm(self, point: tuple) -> str:
        """
        Get the Universal Traverse Mercator projection code for the given area
        :param point: sample point to extract the UTM code from
        :return: Universal Traverse Mercator EPSG code for the given area
        """
        utm_zone = utm.from_latlon(*point)
        if utm_zone[3] > 'N':
            epsg = '326'
        else:
            epsg = '327'
        return epsg + str(utm_zone[2])

    def change_coordinates_to_utm(self, agg_data: pd.DataFrame) -> gpd.GeoDataFrame:
        '''
        Convert the coordinates to Universal Traverse Mercaptor coordinates, which allow the calculating distances to meters without haversine calculation.
        Method detects UTM zone and converts the coordinates accordingly

        :param agg_data: coordinates in WGS84
        :return: AIS dataframe with added UTM coordinate columns
        '''
        gdf = gpd.GeoDataFrame(agg_data, geometry=gpd.points_from_xy(agg_data.lng, agg_data.lat),
                               crs='epsg:4326')
        self.utm_zone = self._get_utm(gdf.iloc[0][['lat', 'lng']].values)
        gdf.to_crs(f'epsg:{self.utm_zone}', inplace=True)
        gdf['lon_utm'] = gdf.geometry.x
        gdf['lat_utm'] = gdf.geometry.y
        return gdf

    def train_model(self):
        """
        Trains the clustering model and assigns the cluster memberships to the train data. Then does aggregate of the clusters
        where each point is represented by the max values of the group (i.e. ship dimensions)
        """
        model = self.algorithm(min_samples=2, eps=20)
        model.fit(self.train_data[['lon_utm', 'lat_utm']].values)
        self.train_data['cluster'] = model.labels_
        self.train_data['trueheading'] = self.train_data.trueheading.astype(float)
        self.clustering_results = self.train_data.groupby('cluster').max()
        self.clustering_results['trueheading'] = self.train_data.groupby('cluster').trueheading.apply(circmean,
                                                                                                      high=360)
        self.clustering_results = gpd.GeoDataFrame(self.clustering_results,
                                                   geometry=gpd.points_from_xy(self.clustering_results.lng,
                                                                               self.clustering_results.lat, crs=4326))

    def create_auxilliary_point(self, point: Point, angle: float, d: float) -> Point:
        """
        Create an auxiliary point that is projected from point based on bearing and distance
        :param point: start geometry point
        :param angle: angle of projection
        :param d: distance of projection
        :return: projected point
        """
        alpha = np.radians(angle)
        xx = point.x + (d * np.sin(alpha))
        yy = point.y + (d * np.cos(alpha))
        return Point([xx, yy])

    def create_polygons(self):
        """
        Creates polygons based on clustering results. The dimensions of the polygon are gained from the max values of
        ships in the group.
        """
        self.clustering_results = self.clustering_results.to_crs(self.utm_zone)
        self.clustering_results['pointA'] = self.clustering_results.apply(
            lambda x: self.create_auxilliary_point(x.geometry, x.trueheading, x.tobow),
            axis=1)
        self.clustering_results['pointB'] = self.clustering_results.apply(
            lambda x: self.create_auxilliary_point(x.geometry, (180 + x.trueheading) % 360, x.tostern), axis=1)
        self.clustering_results['line'] = self.clustering_results.apply(lambda x: LineString([x.pointA, x.pointB]),
                                                                        axis=1)
        self.clustering_results['polygonA'] = self.clustering_results.apply(
            lambda x: x.line.buffer(x.tostarboard, single_sided=True), axis=1)
        self.clustering_results['polygonB'] = self.clustering_results.apply(
            lambda x: x.line.buffer(-x.toport, single_sided=True), axis=1)
        self.clustering_results.polygonA = self.clustering_results.polygonA.set_crs(self.utm_zone)
        self.clustering_results.polygonA = self.clustering_results.polygonA.to_crs(4326)
        self.clustering_results.polygonB = self.clustering_results.polygonB.set_crs(self.utm_zone)
        self.clustering_results.polygonB = self.clustering_results.polygonA.to_crs(4326)
        self.clustering_results = self.clustering_results.to_crs(4326)
        self.clustering_results['geometry'] = self.clustering_results.apply(
            lambda x: unary_union([x.polygonA, x.polygonB]), axis=1)

    def filter_polygons(self):
        """
        Filters out overlapping polygons from the data as well as removes the outlier polygon group.
        """
        self.clustering_results = self.clustering_results[self.clustering_results.geometry.apply(lambda x:
                                                                                                 self._small_intersection(
                                                                                                     x))]
        self.clustering_results = self.clustering_results.loc[self.clustering_results.index.difference([-1]), :]

    def _small_intersection(self, x: Polygon) -> bool:
        """
        Check whether a polygon has significant overlap with another polygon. If the polygon is smaller,
        remove it from the dataframe.
        :param x: Polygon to be checked
        :return: whether there is significant overlap
        """
        for geo in self.clustering_results.geometry.values:
            if geo.intersects(x):
                if geo.intersection(x).area > x.area * 0.25 and geo.area > x.area:
                    return False
        return True

    def _combine_collinear_points(self):
        """
        Combines all berth polygons to quay polygons. A quay is detected if the polygons are collinear and not too far
        away from each other
        :return: quay groups
        """
        groups = []
        df = self.clustering_results.copy()
        df.to_crs(self.utm_zone, inplace=True)
        while len(df) > 0:
            median_angle = df.trueheading.quantile(interpolation='lower')
            med_row = df[df.trueheading == median_angle]
            median_point = med_row.iloc[0].pointA
            group = df[df.apply(lambda x: self._absolute_angle_difference(median_angle, x.trueheading) < 10 and
                                          self._check_collinearity(x.pointA, x.pointB, median_point), axis=1)]
            groups.append(group)
            df.drop(index=group.index, inplace=True)
        return groups

    def _check_collinearity(self, center: Point, projection: Point, point: Point) -> bool:
        """
        Check if two polygons are collinear. Collinearity is checked by taking center point and projection point from
        the reference polygon and these are compared to the center point of the polygon that is tested. The determinant
        (or area) is calculated from these three points and if the area (when also taking the distance between these
         polygons into account) is small enough, the polygons are considered collinear. Also if the distance between the
         two polygons is large enough, they are not concidered as part of the same quay.
        :param center: Center point of polygon A
        :param projection: Projection point of polygon A, created by the create_auxilliary_point ,method
        :param point: Center point of polygon B
        :return: boolean whether the tested polygons are collinear
        """
        a = np.array([[center.x, center.y, 1], [projection.x, projection.y, 1], [point.x, point.y, 1]])
        area = 0.5 * np.linalg.det(a)
        distance = center.distance(point)
        if distance == 0:
            return True
        if distance > 1000:
            return False
        return abs(area) / distance < 10e-1

    def _absolute_angle_difference(self, target: float, source: float) -> float:
        """
        Calculate the absolute angle difference, so that opposite angles are considered equal. I.e. the absolute angle
        between 270 and 90 is 0 or the absolute angle between 5 and 190 is 15.
        :param target: target angle
        :param source: source angle
        :return: the absolute angle difference
        """
        a = target - source
        a = np.abs((a + 180) % 360 - 180)
        b = target - source - 180
        b = np.abs((b + 180) % 360 - 180)
        return min(a, b)

    def find_quays(self):
        """
        Combines found berth polygons to quays, where a quay is formed from one or more collinear berth polygons.
        """
        q = self._combine_collinear_points()
        quay_polys = []
        size = []
        for quay in q:
            multipolys = []
            for _, row in quay.iterrows():
                multipolys.append(row.geometry)
            quay_polys.append(MultiPolygon(multipolys).convex_hull)
            size.append(len(quay))
        self.combined_quays = gpd.GeoDataFrame(data={'size': size}, geometry=quay_polys,
                                               columns=['size'], crs=self.utm_zone)
        self.combined_quays.to_crs(4326, inplace=True)


class Visualizer:
    def __init__(self, data):
        self.data = data
        self.fmap = self.create_map()
        self.map_quays()
        self.map_berths()
        self.save_map()

    def create_map(self):
        """
        Creates a folium map and adds a satellite TileLayer. The map automatically zooms into a location where the data
        is located
        :return:
        """
        m = folium.Map(location=(self.data.train_data.iloc[0][['lat', 'lng']].values), zoom_start=10)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        return m

    def map_berths(self):
        """
        Map the berth polygons
        """
        folium.Choropleth(
            geo_data=self.data.clustering_results[['uid', 'geometry']].to_json(),
            name='Berths',
            fill_color='PuOr',
            fill_opacity=0,
            line_color='blue',
            line_weight=2,
        ).add_to(self.fmap)

    def map_quays(self):
        """
        Map the quay polygons after doing a small buffer on them
        """
        gdata = self.data.combined_quays.copy()
        gdata.geometry = gdata.buffer(0.0002, cap_style='square')
        folium.Choropleth(
            geo_data=gdata.to_json(),
            name='Quays',
            fill_color=None,
            fill_opacity=0,
            line_color='green',
            line_weight=2,
        ).add_to(self.fmap)

    def save_map(self):
        """
        Save the map to a html file
        :param filename: name of file
        """
        folium.LayerControl(collapsed=False).add_to(self.fmap)
        self.fmap.save(cfg.HTML_FILE)


if __name__ == '__main__':
    data = DataObject()
    data.run()
    ml_model = Model()
    ml_model.run(data.train)
    viz = Visualizer(ml_model)
