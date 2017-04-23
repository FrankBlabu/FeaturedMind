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
        
    def add_caption (self, text):
        pass
            
    def add_table (self, rows):
        pass
    
    def add_content (self, element, content):
        pass

    def add_image (self, image, element=None):
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

    css = '\
    .image { \
        width: 100%; \
        height: 100%; \
    } \
    \
    .image img { \
        -webkit-transition: all 1s ease; /* Safari and Chrome */ \
        -moz-transition: all 1s ease; /* Firefox */ \
        -ms-transition: all 1s ease; /* IE 9 */ \
        -o-transition: all 1s ease; /* Opera */ \
        transition: all 1s ease; \
    } \
    \
    .image:hover img { \
        -webkit-transform:scale(2.0); /* Safari and Chrome */ \
        -moz-transform:scale(2.0); /* Firefox */ \
        -ms-transform:scale(2.0); /* IE 9 */ \
        -o-transform:scale(2.0); /* Opera */ \
        transform:scale(2.0); \
    }'

    def __init__ (self, directory, title):
        '''
        Initialize logger
        
        @param directory Directory the log and the resources files are written
                         into. If not existing, the directory will be created.
        @param title     Log title
        '''
        
        self.directory = os.path.abspath (directory)
        self.resource_counter = 0
        
        if not os.path.exists (self.directory):
            os.makedirs (self.directory)
        
        self.document = ET.Element ('html')
        head = ET.SubElement (self.document, 'head')

        ET.SubElement (head, 'meta',  {'charset': 'utf-8'})
        
        if title:
            title_tag = ET.SubElement (head, 'title')
            title_tag.text = title
        
        style = ET.SubElement (head, 'style')
        style.text = HTMLLogger.css
        
        self.body = ET.SubElement (self.document, 'body')
        
        if title:
            caption= ET.SubElement (self.body, 'h1')
            caption.text = title

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
        
    def add_caption (self, text):
        '''
        Add caption to document body
        
        @param text Caption text to add
        '''
        
        caption = ET.SubElement (self.body, 'h2')
        caption.text = caption
            
    def add_table (self, rows, has_header=False):
        '''
        Add HTML table to log
        
        @param rows       Table rows
        @param has_header If 'True', the first row is interpreted as the table header
        '''
        
        table = ET.SubElement (self.body, 'table')        
        
        row_count = 0
        for row in rows:
            tr = ET.SubElement (table, 'tr')
        
            for cell in row:
                if not has_header or row_count > 0:
                    td = ET.SubElement (tr, 'td')
                else: 
                    td = ET.SubElement (tr, 'th')
                    
                self.add_content (td, cell)
                
            row_count += 1 
    
    
    def add_content (self, element, content):
        '''
        Add content (text, image, ...) to document
        
        @param element Element the content is added to
        @param content Content to be added. Can be a string or a numpy array representing an image.
        '''        
        
        if isinstance (content, str):
            element.text = content
            
        elif isinstance (content, np.ndarray):
            self.add_image (content, element=element)
            
        else:
            raise Exception ("Unsupported data type '{0}'".format (type (content))) 
        

    def add_image (self, image, element=None):
        '''
        Add image to document
        
        @param image   Image to add (numpy array format)
        @param element Element the image is added to (body by default)
        '''

        if element is None:
            element = self.body

        file = 'image_{0}.png'.format (self.resource_counter)
        self.resource_counter += 1

        div1 = ET.SubElement (element, 'div', {'class': 'thumbnail'})
        div2 = ET.SubElement (div1, 'div', {'class': 'image'})
            
        ET.SubElement (div2, 'img', {'src': './' + file})

        with warnings.catch_warnings ():
            warnings.simplefilter ('ignore')
            skimage.io.imsave (os.path.join (self.directory, file), skimage.img_as_uint (image))
                
                

