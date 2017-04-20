#!/usr/bin/python3
#
# log.py - Logging functions
#
# Frank Blankenburg, Apr. 2017
#

import os.path
import warnings
import xml.etree.ElementTree as ET
import xml.dom.minidom

import numpy as np

import skimage.color
import skimage.io


#--------------------------------------------------------------------------
# CLASS NoLogger
#
class NoLogger:
    '''
    Logger which does not log anything. Meant to be a stub for 'with'
    statements.
    '''

    def __init__ (self):
        pass
    
    def __enter__ (self):
        return self
    
    def __exit__ (self, exc_type, exc_value, exc_tb):
        pass
            
    def add_row (self, columns):
        pass
            
#--------------------------------------------------------------------------
# CLASS HTMLLogger
#
class HTMLLogger:
    '''
    This class is used to generate HTML based logs. It can list strings and
    images in a HTML table and add various resources to a log directory so 
    that the resultin HTML page is complete and well formatted.
    '''

    def __init__ (self, directory, title, header):
        '''
        Initialize logger
        
        @param directory Directory the log and the resources files are written
                         into. If not existing, the directory will be created.
        @param title     Log title
        @param header    List of header cells
        '''
        
        self.directory = os.path.abspath (directory)
        self.header = header
        self.resource_counter = 0
        
        if not os.path.exists (self.directory):
            os.makedirs (self.directory)
        
        self.document = ET.Element ('html')
        head = ET.SubElement (self.document, 'head')
        
        if title:
            title_tag = ET.SubElement (head, 'title')
            title_tag.text = title
        
        body = ET.SubElement (self.document, 'body')
        
        if title:
            caption= ET.SubElement (body, 'h1')
            caption.text = title

        self.table = ET.SubElement (body, 'table')
        tr = ET.SubElement (self.table, 'tr')
        
        for cell in header:
            self.add_table_cell (tr, cell, tag='th') 

    def __enter__ (self):
        return self
    
    def __exit__ (self, exc_type, exc_value, exc_tb):
        '''
        Exit 'with' scope. The document is written in this case.
        '''
        if self.document:
            file = open (os.path.join (self.directory, 'index.html'), 'w')
        
            text = ET.tostring (self.document, 'utf-8')
            file.write (xml.dom.minidom.parseString (text).toprettyxml (indent='  '))
            
            file.flush ()
            file.close ()
    
    def add_row (self, columns):
        '''
        Add a row to the logging table
        
        @param columns Ordered list of columns to be added. The number of columns must match
                       the number of columns of the header row. 
        '''
        assert len (columns) == len (self.header)
        
        tr = ET.SubElement (self.table, 'tr')
        
        for cell in columns:
            self.add_table_cell (tr, cell)
            
    def add_table_cell (self, row, entry, tag='td'):
        '''
        Generate a table cell
        
        @param row   Parent row element the cell is to be added to
        @param entry Cell entry. Can be a string or a numpy array representing an image.
        @param tag   HTML tag to be used for the cell. Default is 'td'
        '''        
        cell = ET.SubElement (row, tag)
        
        if isinstance (entry, str):
            cell.text = entry
        elif isinstance (entry, np.ndarray):
            file = 'image_{0}.png'.format (self.resource_counter)
            self.resource_counter += 1
            
            ET.SubElement (cell, 'img', {'src': './' + file})
            
            if len (entry.shape) == 3:
                entry = entry.reshape ((entry.shape[0], entry.shape[1]))
            
            image = skimage.color.gray2rgb (entry)
            
            with warnings.catch_warnings ():
                warnings.simplefilter ('ignore')
                skimage.io.imsave (os.path.join (self.directory, file), skimage.img_as_uint (image))
            
        else:
            raise Exception ("Unsupported data type '{0}'".format (type (entry))) 
        

